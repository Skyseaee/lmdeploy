#!/usr/bin/env python3
"""
Run one request with lmdeploy and make sure something is printed.
Supports TurboMind or PyTorch engine.
"""

import argparse
import asyncio
import os
from typing import List

from lmdeploy.messages import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.tokenizer import Tokenizer
from lmdeploy.utils import get_logger

get_logger('lmdeploy').setLevel('DEBUG')

async def run_once(model, tokenizer, prompt_ids: List[int], gen_cfg: GenerationConfig):
    chatbot = model.create_instance()
    session_id = 0
    new_ids: List[int] = []

    async for out in chatbot.async_stream_infer(
            session_id,
            input_ids=prompt_ids,
            gen_config=gen_cfg,
            sequence_start=True,
            sequence_end=True,
            stream_output=True):
        
        # ① 只关心新出的 token 数
        new_n = out.num_token         # 本轮新增 token 数
        if new_n == 0:
            continue

        # ② 取增量 id 并手动解码
        new_ids = out.token_ids

    # 兼容 PyTorch Engine 手动结束
    if hasattr(chatbot, "end"):
        await chatbot.async_end(session_id)
        
    print(new_ids)

    # ---- 非流式兜底打印 ----
    if new_ids:
        # 去掉 prompt 部分，只解码新生成的 token
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        print("\n\n==== decoded once more ====\n" + text)


def build_engine(path: str, backend: str, session_len: int, tp: int, dtype: str, tok):
    if backend == "turbomind":
        from lmdeploy.turbomind import TurboMind
        cfg = TurbomindEngineConfig(tp=tp, session_len=session_len, dtype=dtype)
        return TurboMind.from_pretrained(path, tokenizer=tok, engine_config=cfg)
    else:
        from lmdeploy.pytorch.engine import Engine
        cfg = PytorchEngineConfig(tp=tp, session_len=session_len, dtype=dtype)
        return Engine(path, tokenizer=tok, engine_config=cfg)


def parse_args():
    p = argparse.ArgumentParser("minimal lmdeploy run")
    p.add_argument("model_path")
    p.add_argument("--backend", choices=["turbomind", "pytorch"], default="turbomind")
    p.add_argument("--prompt", default="Hello, how are you?")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--dtype", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = Tokenizer(args.model_path)
    prompt_ids = tokenizer.encode(args.prompt)

    gen_cfg = GenerationConfig(max_new_tokens=args.max_new_tokens, ignore_eos=True)
    sess_len = len(prompt_ids) + args.max_new_tokens

    model = build_engine(args.model_path, args.backend, sess_len, args.tp, args.dtype, tokenizer)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_once(model, tokenizer, prompt_ids, gen_cfg))

    print("\n\n==== end main function ====\n")

    model.close()


if __name__ == "__main__":
    main()