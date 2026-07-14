import os
import numpy as np
import copy
from verl.workers.agent.tool_envs import ToolBase
from typing import Optional, List, Dict, Any
from PIL import Image
import re
import json
from verl.workers.agent.envs.mm_process_engine.prompt import PROMPT
from math import ceil, floor
# 临时修复
# ToolBase.registry = {}


# ===================== 坐标空间配置 =====================
# Qwen2.5-VL : bbox_2d 输出的是（原图）像素坐标      -> "pixel"
# Qwen3-VL   : bbox_2d 输出的是 0-1000 归一化坐标    -> "qwen3"
#
# 通过环境变量 BBOX_COORD_SPACE 切换，默认 qwen3。
# 想切回 Qwen2.5-VL 时，在训练脚本里 export BBOX_COORD_SPACE=pixel 即可，
# 不需要改代码。
# =======================================================
BBOX_COORD_SPACE = os.environ.get("BBOX_COORD_SPACE", "qwen3").lower()

# 归一化坐标系的上界（Qwen3-VL 官方为 1000）
QWEN3_COORD_SCALE = float(os.environ.get("BBOX_COORD_SCALE", "1000"))

# crop 之后送回模型的图，最短边至少要有这么多像素，否则 vision encoder 没法处理
MIN_CROP_SIDE = 28

# validate_bbox 里的"太小"判定阈值。
# 注意：这个值必须 < MIN_CROP_SIDE，否则下面 28px 的自动扩张逻辑永远不会被触发。
MIN_VALID_SIDE = 4


class VisualToolBoxV2(ToolBase):
    name = "visual_toolbox_v2"
    # user_prompt = "Here is the cropped image returned after you calling the function {}.\nIf the images provided above are sufficient to answer the user's question, please put your final answer within <answer></answer>. Otherwise you can continue to call tools within <tool_call></tool_call>."

    user_prompt = PROMPT.USER_PROMPT_V2

    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(
            name=self.name,
        )
        self.chatml_history = []
        self.multi_modal_data = None  # To store the current image being processed

    def extract_answer(self, action_string: str) -> Dict[str, any]:
        answer = re.findall(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
        return answer[-1] if answer else None

    def extract_action(self, action_string: str) -> Dict[str, Any]:
        """
        Extracts the tool call from the action string.

        Args:
            action_string: The string containing the tool call in XML tags.

        Returns:
            A dictionary with the tool name and arguments.

        Raises:
            ValueError: If no tool call is found or JSON is invalid.
        """
        tool_call_match = re.findall(r'<tool_call>(.*?)</tool_call>', action_string, re.DOTALL)
        return tool_call_match[-1] if tool_call_match else None

    def execute(self, action_string: str, **kwargs) -> tuple:
        """
        Execute the tool functionality based on the action string.

        Args:
            action_string: The string containing the tool call in XML tags.

        Returns:
            observation: The structured observation with the processed image.
            reward: 0.1 if tool call is successful with correct JSON format, 0 otherwise.
            done: Whether the episode is terminated.
            info: Additional info.
        """

        answer = self.extract_answer(action_string)
        if answer:
            return "", 0.0, True, {}
        action = self.extract_action(action_string)
        if not action:
            return "", 0.0, True, {}
        try:
            tool_call = json.loads(action.strip())  # 或使用 literal_eval
        except Exception as e:
            error_msg = f"Invalid tool call format: {action.strip()}. Error: {e}"
            obs = "\n<|im_start|>user\n" + f"Error: {str(error_msg)}" + "<|im_end|>\n<|im_start|>assistant\n"
            info = {"error": str(e), "status": "failed"}
            return obs, 0.0, False, info
        try:

            tool_name = tool_call["name"]
            args = tool_call["arguments"]

            if tool_name == "image_zoom_in_tool":
                # Zoom in by cropping the image
                # image_path = args["image_path"]
                bbox = args["bbox_2d"]
                bbox = self.maybe_resize_bbox(*bbox)
                if not bbox:
                    raise ValueError(f"ZOOM IN ARGUMENTS ARE INVALID")
                # img = Image.open(image_path)
                img = self.multi_modal_data['image'][0]
                cropped_img = img.crop(bbox)
                current_image = cropped_img

            elif tool_name == "image_rotate_tool":
                # Rotate the image
                # image_path = args["image_path"]
                angle = args["angle"]
                # img = Image.open(image_path)
                img = self.multi_modal_data['image'][0]
                rotated_img = img.rotate(angle)
                current_image = rotated_img

            else:
                raise ValueError(f"Unknown tool name: {tool_name}")
            # Prepare the observation
            obs = {
                "prompt": "\n<|im_start|>user\n" + "<tool_response>" + "<image>" + self.user_prompt + "</tool_response>" + "<|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {"image": [current_image]}
            }
            reward = 0.0  # Reward for successful tool call with correct JSON
            done = False
            info = {"status": "success", "tool_used": tool_name}
            print(f'[DEBUG] SUCCESS ACTION {action_string=}')
            return obs, reward, done, info
        except Exception as e:
            # Return an error observation if something goes wrong
            print(f'[DEBUG] Execute WRONG - {str(e)} {action_string=}')
            obs = "\n<|im_start|>user\n" + f"Error: {str(e)}" + "<|im_end|>\n<|im_start|>assistant\n"
            reward = 0.0  # No reward for failed execution
            done = False
            info = {"error": str(e), "status": "failed"}
            return obs, reward, done, info

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'

        self.height = self.multi_modal_data['image'][0].height
        self.width = self.multi_modal_data['image'][0].width

    # ---------------- 新增：坐标空间换算 ----------------
    def rescale_bbox(self, left, top, right, bottom):
        """把模型输出的 bbox 统一换算到原图像素坐标系。

        Qwen3-VL 输出的是归一化到 [0, 1000] 的坐标，与图像实际分辨率无关；
        而本类的 self.width / self.height 以及后面 img.crop() 用的都是原图
        真实像素。两者不换算直接比较，就会出现 left=519 > right=500 这种
        "看起来越界、其实是单位不同" 的错误。
        """
        if BBOX_COORD_SPACE != "qwen3":
            # Qwen2.5-VL 等：模型直接输出像素坐标，原样返回
            return left, top, right, bottom

        sx = self.width / QWEN3_COORD_SCALE
        sy = self.height / QWEN3_COORD_SCALE

        left = int(round(left * sx))
        top = int(round(top * sy))
        right = int(round(right * sx))
        bottom = int(round(bottom * sy))
        return left, top, right, bottom

    def validate_bbox(self, left, top, right, bottom):
        try:
            assert left < right and bottom > top, f'invalid shape for {left=}, {top=}, {right=}, {bottom=}'
            height = bottom - top
            width = right - left
            assert max(height, width) / min(height, width) <= 100, f"aspect ratio error: {left=}, {top=}, {right=}, {bottom=}"
            # 原来这里写死 30，比下面 28px 的扩张阈值还大，导致小框在扩张前
            # 就被判死，扩张逻辑形同虚设。改成 MIN_VALID_SIDE(=4)。
            assert min(height, width) > MIN_VALID_SIDE, f"{height=}, {width=} is too small"
            return True
        except Exception as err:
            print(f' [ERROR vl_agent #2] {err=}')
            return False

    def maybe_resize_bbox(self, left, top, right, bottom):
        # 1) 先把坐标换算到原图像素空间（Qwen3-VL: 0-1000 -> pixel）
        left, top, right, bottom = self.rescale_bbox(left, top, right, bottom)

        # 2) 模型偶尔会把 x1/x2 或 y1/y2 写反，直接交换而不是判死
        if left > right:
            left, right = right, left
        if top > bottom:
            top, bottom = bottom, top

        # 3) 夹到图像边界内
        left = max(0, left)
        top = max(0, top)
        right = min(self.width, right)
        bottom = min(self.height, bottom)

        # 4) 扩张前只做最基础的"形状没退化成负数/零"检查，
        #    不在这里做尺寸/长宽比校验（那是 validate_bbox 的活）。
        #    这一步和下面完整的 validate_bbox 是两件不同的事：
        #    这里只是过滤 clamp 之后彻底不成形的框（比如 left==right），
        #    不能提前用"太小"这个标准卡掉——否则比 MIN_CROP_SIDE 还小的框
        #    永远走不到下面的扩张逻辑，扩张分支就变成死代码。
        if left >= right or top >= bottom:
            print(f' [ERROR vl_agent #2] degenerate bbox after clamp: {left=}, {top=}, {right=}, {bottom=}')
            return None

        height = bottom - top
        width = right - left
        if height < MIN_CROP_SIDE or width < MIN_CROP_SIDE:
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            # min(height, width) 此时必然 > 0（上面已经拦掉了 == 0 的情况），
            # 但为防御性起见仍然设一个下限，避免除以极小值时 ratio 爆炸。
            ratio = MIN_CROP_SIDE / max(min(height, width), 1.0)
            new_half_height = ceil(height * ratio * 0.5)
            new_half_width = ceil(width * ratio * 0.5)
            left = max(0, floor(center_x - new_half_width))
            right = min(self.width, ceil(center_x + new_half_width))
            top = max(0, floor(center_y - new_half_height))
            bottom = min(self.height, ceil(center_y + new_half_height))

        # 5) 完整校验（长宽比 + 最小边长）只在这里做一次，
        #    此时框已经是"扩张后的最终版本"，不会再被过早拦截。
        if not self.validate_bbox(left, top, right, bottom):
            return None
        return [left, top, right, bottom]


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Part A: 端到端测试 execute()。
    #
    # 之前这里直接照抄了一份示例代码，有三处硬伤，从来没有真正跑通过：
    #   1) 类名写的是 VisualToolBox，这个文件里只定义了 VisualToolBoxV2
    #      -> NameError
    #   2) 测试 JSON 用的是 "bbox" 字段，但 execute() 实际读的是
    #      args["bbox_2d"] -> KeyError
    #   3) 没调用 reset() 就直接 execute()，self.multi_modal_data 还是
    #      None -> TypeError: 'NoneType' object is not subscriptable
    # 下面这版把这三处都修了，用一张内存里生成的假图跑真实的 execute()，
    # 覆盖原本想测的 4 个场景：zoom in / rotate / 非法JSON / 未知工具。
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Part A: end-to-end execute() tests")
    print("=" * 60)

    def make_tool(width=500, height=333):
        """构造一个可用的 VisualToolBoxV2 实例，模拟 reset() 之后的状态。"""
        t = VisualToolBoxV2.__new__(VisualToolBoxV2)
        t.chatml_history = []
        dummy_img = Image.new("RGB", (width, height), color=(120, 180, 220))
        t.multi_modal_data = {"image": [dummy_img]}
        t.height = height
        t.width = width
        return t

    # A1. Zoom in：用 Qwen3-VL 风格的 0-1000 归一化坐标（模拟真实模型输出）
    tool = make_tool()
    zoom_in_action = """
    <tool_call>
    {"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [519, 265, 703, 546], "label": "test region"}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(zoom_in_action)
    print(f"[A1] Zoom in (qwen3 0-1000 coords) -> status={info.get('status')}")
    assert info.get("status") == "success", "A1 应该成功：这正是之前报 invalid shape 的那类坐标"

    # A2. Rotate
    tool = make_tool()
    rotate_action = """
    <tool_call>
    {"name": "image_rotate_tool", "arguments": {"angle": 90}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(rotate_action)
    print(f"[A2] Rotate -> status={info.get('status')}")
    assert info.get("status") == "success", "A2 应该成功"

    # A3. 非法 JSON
    tool = make_tool()
    invalid_action = """
    <tool_call>
    {"name": "image_rotate_tool", "arguments": {"angle": 90}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(invalid_action)
    print(f"[A3] Invalid JSON -> status={info.get('status')}")
    assert info.get("status") == "failed", "A3 应该失败（JSON 本身不合法）"

    # A4. 未知工具名
    tool = make_tool()
    unknown_tool_action = """
    <tool_call>
    {"name": "unknown_tool", "arguments": {"param": "value"}}
    </tool_call>
    """
    obs, reward, done, info = tool.execute(unknown_tool_action)
    print(f"[A4] Unknown tool -> status={info.get('status')}")
    assert info.get("status") == "failed", "A4 应该失败（未知工具名）"

    print("Part A 全部通过\n")

    # ------------------------------------------------------------------
    # Part B: 坐标换算专项回归测试。
    # 每一条都是从真实训练日志里摘出来的 (bbox_2d, 图像宽高) 组合，
    # 宽高是根据当时报错信息里 clamp 后的 right/bottom 反推出来的。
    # 修复前这些全部返回 None（ZOOM IN ARGUMENTS ARE INVALID）。
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"Part B: coordinate rescale regression (BBOX_COORD_SPACE={BBOX_COORD_SPACE})")
    print("=" * 60)

    cases = [
        # (bbox_2d, width, height, label)
        ([697, 479, 820, 578], 500, 375, "house on the right side"),
        ([519, 265, 703, 546], 500, 375, "bush with leaves"),
        ([0, 0, 995, 302], 500, 375, "graph top part (之前判SUCCESS但裁错区域)"),
        ([56, 208, 531, 861], 500, 375, "graph bottom part (之前判SUCCESS但裁错区域)"),
        ([273, 581, 523, 843], 500, 333, "banana on the table"),
        ([785, 212, 929, 356], 500, 333, "cup on a counter"),
        ([673, 237, 934, 943], 640, 489, "man in green jersey"),
        ([212, 497, 386, 585], 386, 375, "individuals in the boat"),
        ([500, 500, 505, 505], 500, 375, "极小框，触发 28px 扩张分支"),
        ([703, 546, 519, 265], 500, 375, "坐标写反，应被自动交换"),
    ]
    all_ok = True
    for bbox, w, h, label in cases:
        t = make_tool(width=w, height=h)
        result = t.maybe_resize_bbox(*bbox)
        ok = result is not None
        all_ok = all_ok and ok
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] bbox={bbox} image={w}x{h} ({label}) -> {result}")

    print()
    if all_ok:
        print("Part B 全部通过：之前会报 invalid shape / 裁错区域的坐标，现在都能正确落回图内。")
    else:
        print("Part B 有失败用例，请检查 rescale_bbox / maybe_resize_bbox 实现。")