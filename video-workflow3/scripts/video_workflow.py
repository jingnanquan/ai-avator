#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI视频生成工作流
功能：根据口播主题/文案，调用Seedance生成视频片段，FFmpeg合并
作者：OpenClaw
"""

import base64
import os
import time
from datetime import datetime

import ffmpeg
import requests
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.chat_models import ChatTongyi
from pydantic_settings import BaseSettings, SettingsConfigDict
from volcenginesdkarkruntime import Ark
import argparse
import sys
import json


# ====================== 1. 配置管理 ======================
class Settings(BaseSettings):
    """全局配置（支持.env文件）"""
    # LLM配置
    llm_provider: str = "qwen"
    qwen_api_key: str = "sk-5bf790b928cf4c75be2a15d205bf8cf8"

    # Seedance API配置
    seedance_api_key: str = "55058e77-f549-4704-8515-10738479eb7e"
    seedance_model: str = "doubao-seedance-1-5-pro-251215"
    seedance_base_url: str = "https://api.seedance.ai/v1"
    seedance_poll_interval: int = 15
    seedance_timeout: int = 300

    # 视频配置
    video_segment_duration: int = 8
    final_video_path: str = "merged_video.mp4"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()


# ====================== 2. 全局客户端（改进2：只初始化一次） ======================
_seedance_client = None


def get_seedance_client():
    """获取Seedance客户端（单例模式）"""
    global _seedance_client
    if _seedance_client is None:
        _seedance_client = Ark(api_key=settings.seedance_api_key)
    return _seedance_client


# ====================== 3. 工作流管理器（改进1：按时间创建文件夹） ======================
class WorkflowManager:
    """管理工作流运行目录"""
    
    def __init__(self):
        self.run_dir: str = ""
        self._create_run_dir()
    
    def _create_run_dir(self):
        """创建以当前时间命名的文件夹"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(os.getcwd(), f"workflow_run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        print(f"📂 工作流输出目录: {self.run_dir}")
    
    def get_output_path(self, filename: str) -> str:
        """获取输出文件路径"""
        return os.path.join(self.run_dir, filename)
    
    def get_video_dir(self) -> str:
        """获取视频片段目录"""
        video_dir = os.path.join(self.run_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        return video_dir


# 全局工作流管理器（延迟初始化，在main中创建）
workflow_manager: Optional[WorkflowManager] = None


# ====================== 4. LLM工厂函数 ======================
def get_llm():
    """工厂函数：根据配置返回LangChain封装的LLM实例"""
    if settings.llm_provider == "qwen":
        return ChatTongyi(
            model="qwen-max",
            api_key=settings.qwen_api_key,
        )
    else:
        raise ValueError(f"不支持的LLM供应商: {settings.llm_provider}")


# ====================== 5. 视频处理工具函数 ======================
def extract_last_frame(video_path: str) -> str:
    """提取视频最后一帧"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
    except Exception as e:
        raise RuntimeError(f"获取视频时长失败: {str(e)}")

    # 使用工作流目录存储尾帧
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_path = workflow_manager.get_output_path(f"{base_name}_last_frame.jpg")

    try:
        (
            ffmpeg
            .input(video_path, ss=duration - 0.1)
            .output(frame_path, vframes=1)
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg处理失败: {e.stderr.decode('utf8')}")

    if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
        print(f"✅ 视频尾帧提取成功: {frame_path}")
        return frame_path
    else:
        raise RuntimeError("尾帧提取失败，输出文件无效")


def process_image_input(image_input: str) -> dict:
    """处理图片输入，返回Seedance API所需的格式"""
    if image_input.startswith(("http://", "https://")):
        return {"image_url": {"url": image_input}, "type": "image_url"}
    elif image_input.startswith("data:image/"):
        return {"image_url": {"url": image_input}, "type": "image_url"}
    else:
        if not os.path.isfile(image_input):
            raise FileNotFoundError(f"本地图片文件不存在：{image_input}")

        with open(image_input, "rb") as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')

        ext = os.path.splitext(image_input)[1].lower().lstrip('.')
        mime_type = {'jpg': 'jpeg', 'jpeg': 'jpeg', 'png': 'png', 'webp': 'webp'}.get(ext, 'png')
        data_uri = f"data:image/{mime_type};base64,{img_base64}"

        return {"image_url": {"url": data_uri}, "type": "image_url"}


def call_seedance_api(segment: str, reference_image: str, video_index: int) -> str:
    """调用Seedance API生成视频（使用全局客户端）"""
    # 使用全局客户端，不再每次创建
    client = get_seedance_client()

    prompt_text = f"""基于这张图片生成一段数字人口播视频，
    1.文案如下：'{segment}'
    2.动作描述如下：人物动作自然，镜头固定不移动
    3.注意事项：不要在视频中添加字幕和图案"""

    try:
        image_content = process_image_input(reference_image)
        print(f"📌 创建Seedance任务，文案：{segment[:20]}...")
        
        create_resp = client.content_generation.tasks.create(
            model=settings.seedance_model,
            content=[
                {"text": prompt_text, "type": "text"},
                image_content
            ],
            resolution="480p"
        )
        task_id = create_resp.id
        print(f"✅ 任务创建成功，ID：{task_id}")

        # 轮询任务状态
        start_time = time.time()
        while True:
            if time.time() - start_time > settings.seedance_timeout:
                raise TimeoutError(f"Seedance任务超时（{settings.seedance_timeout}秒），任务ID：{task_id}")

            task_resp = client.content_generation.tasks.get(task_id=task_id)
            print(f"🔍 任务状态：{task_resp.status}")

            if task_resp.status == "succeeded":
                video_url = task_resp.content.video_url
                break
            elif task_resp.status in ("failed", "expired"):
                raise RuntimeError(f"Seedance任务失败，ID：{task_id}，原因：{getattr(task_resp, 'message', '未知')}")
            else:
                time.sleep(settings.seedance_poll_interval)

        # 下载视频到工作流目录
        video_filename = f"segment_{video_index:03d}_{task_id}.mp4"
        video_path = workflow_manager.get_output_path(video_filename)
        
        # print(f"📥 下载视频：{video_url} → {video_path}")
        download_resp = requests.get(video_url, timeout=60)
        download_resp.raise_for_status()
        
        with open(video_path, "wb") as f:
            f.write(download_resp.content)

        print(f"✅ 视频下载完成：{video_path}")
        return video_path

    except Exception as e:
        raise RuntimeError(f"Seedance API调用失败：{str(e)}")


def ffmpeg_merge(video_paths: List[str], output_filename: str = None) -> str:
    """合并视频片段"""
    if output_filename is None:
        output_filename = settings.final_video_path
    
    output_path = workflow_manager.get_output_path(output_filename)
    
    list_file = workflow_manager.get_output_path("video_list.txt")

    with open(list_file, "w", encoding="utf-8") as f:
        for video_path in video_paths:
            abs_path = os.path.abspath(video_path)
            if os.name == 'nt':
                abs_path = abs_path.replace('\\', '/')
            f.write(f"file '{abs_path}'\n")

    print(f"🔗 将合并 {len(video_paths)} 个视频片段")

    try:
        (
            ffmpeg
            .input(list_file, f='concat', safe=0)
            .output(output_path, c='copy')
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        print("⚠️ 标准concat方法失败，尝试使用filter_complex...")
        try:
            inputs = [ffmpeg.input(vp) for vp in video_paths]
            concat_stream = ffmpeg.concat(*inputs, v=1, a=1)
            (
                ffmpeg
                .output(concat_stream, output_path)
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e2:
            raise RuntimeError(f"FFmpeg合并失败: {e2.stderr.decode('utf8')}")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ 视频合并成功: {output_path} ({file_size_mb:.2f} MB)")
    else:
        raise RuntimeError("合并后的视频文件无效或为空")
    
    return output_path


# ====================== 6. LangGraph工作流核心 ======================
class WorkflowState(TypedDict):
    """工作流状态定义"""
    topic: Optional[str]
    reference_image: str
    script: Optional[str]
    segments: Optional[List[str]]
    video_paths: Optional[List[str]]
    final_video: Optional[str]
    start_step: str = "generate_script"
    # 交互式执行：当下一步等于 end_step 时提前结束
    end_step: Optional[str]


def generate_script_node(state: WorkflowState) -> WorkflowState:
    """生成口播文案节点"""
    if not state.get("topic"):
        raise ValueError("生成文案必须提供topic参数")

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("user", f"""你是一名专业的口播导演，请根据主题生成口播文案，要求：
1. 字数40-60字，自然流畅，中文
2. 符合口播语气，适合视频播放
3. 主题：{state['topic']}
仅返回文案内容，无需额外解释。""")
    ])

    chain = prompt | llm | StrOutputParser()
    script = chain.invoke({"topic": state["topic"]})

    print(f"✅ 生成口播文案：\n{script[:100]}...")
    return {"script": script}


def split_script_node(state: WorkflowState) -> WorkflowState:
    """拆分文案节点"""
    script = state.get("script")
    if not script:
        raise ValueError("拆分文案必须提供script参数")

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("user", f"""将以下口播文案拆分为{settings.video_segment_duration}秒/段的片段（每段50-80字,3~4句话最佳）：
{script}
要求：
1. 拆分后的片段符合口播节奏，每段独立成意
2. 仅返回JSON格式的列表，不要添加任何额外解释或说明。示例：["片段1", "片段2"]
3. 不要对口播文案进行任何修改。
4. 对输入的文本较短的句子，不用拆成多段。

案例：
口播文案1："数字人技术正深度变革我们的数字生活！它能够塑造栩栩如生的虚拟形象，实现7×24小时不间断的智能交互，不仅应用于虚拟主播、数字员工，更能赋能直播带货、品牌营销、在线教育等多个领域，帮助企业降本增效，开拓全新的增长空间。"
输出1：["数字人技术正深度变革我们的数字生活！它能够塑造栩栩如生的虚拟形象，实现7×24小时不间断的智能交互，不仅应用于虚拟主播、数字员工，更能赋能直播带货、品牌营销、在线教育等多个领域，帮助企业降本增效，开拓全新的增长空间。"]

口播文案2："三月江城，春色满园。素有"中国最美大学"之称的武汉大学，近日迎来了2026年度樱花盛花期。珞珈山下，樱花大道两侧的千株樱花竞相绽放，粉白交织的花瓣随风摇曳，与古朴典雅的老斋舍、琉璃碧瓦的老图书馆交相辉映，吸引了来自全国各地的游客前来观赏。据武汉大学发布的通告，今年樱花开放期间（3月13日至31日），校园将继续实行实名预约、免费限流入校的政策。工作日每日预约限额为2万人，周末限额为4万人。游客可通过武汉大学官方网站或官方微信公众号提前预约，预约成功后凭身份证核验入校。"
输出2：["三月江城，春色满园。素有"中国最美大学"之称的武汉大学，近日迎来了2026年度樱花盛花期。珞珈山下，樱花大道两侧的千株樱花竞相绽放，粉白交织的花瓣随风摇曳，与古朴典雅的老斋舍、琉璃碧瓦的老图书馆交相辉映，吸引了来自全国各地的游客前来观赏。","据武汉大学发布的通告，今年樱花开放期间（3月13日至31日），校园将继续实行实名预约、免费限流入校的政策。工作日每日预约限额为2万人，周末限额为4万人。游客可通过武汉大学官方网站或官方微信公众号提前预约，预约成功后凭身份证核验入校。"]

""")
    ])

    chain = prompt | llm | JsonOutputParser()
    segments = chain.invoke({"script": script, "duration": settings.video_segment_duration})

    print(f"✅ 拆分文案为 {len(segments)} 个片段：{segments}")
    return {"segments": segments}


def generate_videos_node(state: WorkflowState) -> WorkflowState:
    """生成视频节点"""
    if not state.get("segments"):
        raise ValueError("生成视频必须提供segments参数")

    video_paths = []
    prev_frame = state["reference_image"]
    segments = state["segments"]

    for i, seg in enumerate(segments):
        print(f"🔄 生成第{i + 1}/{len(segments)}个视频片段")
        
        # 传递索引用于命名
        video_path = call_seedance_api(seg, prev_frame, video_index=i + 1)
        video_paths.append(video_path)
        
        # 更新尾帧
        prev_frame = extract_last_frame(video_path)

    return {"video_paths": video_paths}


def parse_director_segments(director: str) -> List[str]:
    """
    解析 --director 输入为 segments 列表。
    支持：
    - JSON 数组：["片段1","片段2"]
    - 换行/竖线分隔：每行/每项为一个片段
    - 否则作为单段
    """
    director = (director or "").strip()
    if not director:
        raise ValueError("director 不能为空")

    # JSON: ["片段1", "片段2"]
    if director.startswith("["):
        try:
            parsed = json.loads(director)
        except Exception as e:
            raise ValueError(f"director 解析为 JSON 列表失败：{e}")
        if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
            raise ValueError("director JSON 必须是字符串列表，例如：['片段1','片段2']")
        segments = [x.strip() for x in parsed if x and x.strip()]
        if not segments:
            raise ValueError("director JSON 列表解析后为空")
        return segments

    # 换行优先
    lines = [line.strip() for line in director.splitlines() if line.strip()]
    if len(lines) >= 2:
        return lines

    # 竖线拆分
    if "|" in director:
        parts = [p.strip() for p in director.split("|") if p.strip()]
        if parts:
            return parts

    # 兜底：单段
    return [director]


def merge_videos_node(state: WorkflowState) -> WorkflowState:
    """合并视频节点"""
    if not state.get("video_paths"):
        raise ValueError("合并视频必须提供video_paths参数")

    # 使用时间戳命名最终视频
    final_filename = f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    final_video = ffmpeg_merge(state["video_paths"], output_filename=final_filename)
    
    print(f"✅ 视频合并完成：{final_video}")
    return {"final_video": final_video}


def route_start(state: WorkflowState) -> str:
    """路由到指定的执行起点"""
    return state["start_step"]

def route_after_generate_script(state: WorkflowState) -> str:
    """generate_script 执行后：若下一步是 split_script，则提前结束"""
    if state.get("end_step") == "split_script":
        return END
    return "split_script"


def route_after_split_script(state: WorkflowState) -> str:
    """split_script 执行后：若下一步是 generate_videos，则提前结束"""
    if state.get("end_step") == "generate_videos":
        return END
    return "generate_videos"


def route_after_generate_videos(state: WorkflowState) -> str:
    """generate_videos 执行后：若下一步是 merge_videos，则提前结束"""
    if state.get("end_step") == "merge_videos":
        return END
    return "merge_videos"


def build_workflow():
    """构建LangGraph工作流"""
    graph = StateGraph(WorkflowState)

    graph.add_node("generate_script", generate_script_node)
    graph.add_node("split_script", split_script_node)
    graph.add_node("generate_videos", generate_videos_node)
    graph.add_node("merge_videos", merge_videos_node)

    # 条件路由：支持从不同起点开始
    graph.add_conditional_edges(
        START,
        route_start,
        {
            "generate_script": "generate_script",
            "split_script": "split_script",
            "generate_videos": "generate_videos",
        }
    )

    # 节点间交互式流程：当下一步等于 end_step 时提前结束
    graph.add_conditional_edges(
        "generate_script",
        route_after_generate_script,
        {
            "split_script": "split_script",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "split_script",
        route_after_split_script,
        {
            "generate_videos": "generate_videos",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "generate_videos",
        route_after_generate_videos,
        {
            "merge_videos": "merge_videos",
            END: END,
        },
    )
    graph.add_edge("merge_videos", END)

    return graph.compile()


# ====================== 7. 输入验证（改进3） ======================
def validate_inputs(
    topic: str = None,
    script: str = None,
    director: str = None,
    reference_image: str = None,
) -> bool:
    """
    验证输入参数（改进3：入口处检查）
    返回: True=通过, False=不通过
    """
    errors = []
    
    # 检查参考图片（必填）
    if not reference_image:
        errors.append("参考图片不能为空")
    elif not os.path.exists(reference_image):
        errors.append(f"参考图片文件不存在: {reference_image}")
    
    topic_provided = bool(topic and topic.strip())
    script_provided = bool(script and script.strip())
    director_provided = bool(director and director.strip())

    # 只能三者之一
    provided_count = int(topic_provided) + int(script_provided) + int(director_provided)
    if provided_count != 1:
        errors.append("必须且只能提供 --topic(-t)、--script(-s)、--director(-d) 其中一个")
    
    if errors:
        print("❌ 输入参数验证失败：")
        for err in errors:
            print(f"   - {err}")
        return False
    
    return True


# ====================== 8. 主程序入口 ======================
def main(
    topic: str = None,
    script: str = None,
    director: str = None,
    reference_image: str = None,
    end_step: Optional[str] = None,
):
    """
    主函数
    
    Args:
        topic: 口播主题（可选，与script二选一）
        script: 口播文案（可选，与topic二选一）
        director: 导演脚本/分段脚本（可选，与topic/script二选一）
        reference_image: 参考图片路径（必填）
        end_step: 交互式结束点：当下一步等于该值时提前结束（例如 split_script / generate_videos / merge_videos）
    """
    global workflow_manager
    
    # ========== 步骤1: 输入验证 ==========
    print("=" * 50)
    print("步骤1: 输入验证")
    print("=" * 50)
    
    if not validate_inputs(topic=topic, script=script, director=director, reference_image=reference_image):
        print("\n❌ 工作流终止，请检查输入参数")
        return None
    
    print("✅ 输入验证通过\n")

    if director:
        # ========== 步骤x: 初始化工作流目录 ==========
        print("=" * 50)
        print("步骤x: 初始化工作流目录")
        print("=" * 50)
        workflow_manager = WorkflowManager()
    
        # 复制参考图片到工作流目录（备份）
        import shutil
        ref_basename = os.path.basename(reference_image)
        ref_copy = workflow_manager.get_output_path(f"reference_{ref_basename}")
        shutil.copy2(reference_image, ref_copy)
        print(f"✅ 参考图片已备份到: {ref_copy}\n")
    
    # ========== 步骤2: 编译并运行工作流 ==========
    print("=" * 50)
    print("步骤2: 执行工作流")
    print("=" * 50)
    
    app = build_workflow()

    # 根据是否提供输入决定起始步骤
    if topic:
        start_step = "generate_script"
        initial_state = {
            "topic": topic,
            "reference_image": reference_image,
            "start_step": start_step,
        }
        print("📌 模式: topic -> generate_script（交互式：默认生成完文案后结束）")
        if end_step is None:
            end_step = "split_script"

    elif script:
        start_step = "split_script"
        initial_state = {
            "script": script,
            "reference_image": reference_image,
            "start_step": start_step,
        }
        print("📌 模式: script -> split_script（交互式：默认拆分完片段后结束）")
        if end_step is None:
            end_step = "generate_videos"

    else:
        # director：直接进入 generate_videos（要求 director 可解析为 segments）
        start_step = "generate_videos"
        initial_state = {
            "segments": parse_director_segments(director),
            "reference_image": reference_image,
            "start_step": start_step,
        }
        print("📌 模式: director -> generate_videos（默认不在中途停止）")
        # director 默认跑完整个工作流（不设置 end_step）
        if end_step is None:
            end_step = None
    
    print()
    initial_state["end_step"] = end_step
    
    try:
        result = app.invoke(initial_state)
        
        print("\n" + "=" * 50)
        print("✅ 工作流执行完成!")
        print("=" * 50)
        if director:
            print(f"📁 输出目录: {workflow_manager.run_dir}")
            print(f"🎬 最终视频: {result['final_video']}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ 工作流执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ====================== 9. 命令行入口 ======================
if __name__ == "__main__":
# 创建参数解析器
    parser = argparse.ArgumentParser(
        description="处理视频生成的参数解析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""使用示例:
  1. 通过主题生成: python video_workflow.py -i "C:\\1.png" -t "介绍人工智能专业"
  2. 通过文案生成: python video_workflow.py -i "C:\\1.png" -s "视频生成技术已经悄悄进入人们的生活....."
  3. 通过导演/分段直接生成视频: python video_workflow.py -i "C:\\1.png" -d '["片段1","片段2"]'
        """
    )
    
    # 添加命令行参数
    parser.add_argument(
        '-i', '--img', 
        required=True,
        help='参考图片路径 (必填参数)'
    )
    parser.add_argument(
        '-t', '--topic', 
        help='生成视频的主题（与--script/--director二选一）'
    )
    parser.add_argument(
        '-s', '--script', 
        help='视频文案内容（与--topic/--director二选一）'
    )
    parser.add_argument(
        '-d', '--director',
        help='导演脚本/分段脚本（JSON 数组如 ["片段1","片段2"]，或多行/用|分隔的片段；与--topic/--script二选一）',
    )
    parser.add_argument(
        '--end-step',
        dest="end_step",
        default=None,
        help='交互式结束点：当下一步等于该值时提前结束（例如 split_script / generate_videos / merge_videos）。',
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 参数合法性校验
    # 校验1: 检查img参数是否为空
    if not args.img.strip():
        print("错误: 参考图片路径(--img/-i)不能为空！", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    # 校验2: 检查 topic/script/director 的互斥性（必须且只能提供一个）
    topic_provided = args.topic is not None and args.topic.strip() != ""
    script_provided = args.script is not None and args.script.strip() != ""
    director_provided = args.director is not None and args.director.strip() != ""

    provided_count = int(topic_provided) + int(script_provided) + int(director_provided)
    if provided_count != 1:
        print("错误: 必须且只能提供 --topic(-t) / --script(-s) / --director(-d) 其中一个参数！", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    # 调用主函数
    result = main(
        topic=args.topic if topic_provided else None,
        script=args.script if script_provided else None,
        director=args.director if director_provided else None,
        reference_image=args.img,
        end_step=args.end_step,
    )

    if topic_provided:
        res = {"script": result.get("script", "生成为空，可能出现错误。")}
        print(f"文案脚本: { res }")
    elif script_provided:
        res = {"segments": result.get("segments", "生成为空，可能出现错误。")}
        print(f"导演脚本: { res }")
    elif director_provided:
        print(f"执行结果: {result}")
