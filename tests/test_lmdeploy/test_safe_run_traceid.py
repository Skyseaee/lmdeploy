################################################################################
# @Copyright: 2019-2025 Shopee. All Rights Reserved.
# @Author   : zhen.wan@shopee.com
# @Date     : 2025-11-27 15:45:15
# @Details  : Verify that when safe_run() throws an error, the trace ID is
#             correctly returned in both the logs and the error response.
################################################################################
import pytest
import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from lmdeploy.serve.async_engine import AsyncEngine


@pytest.mark.asyncio
async def test_safe_run_error_with_traceid():

    test_traceid = "traceid_123"
    test_session_id = 1

    # Mock logger
    with patch('lmdeploy.serve.async_engine.logger') as mock_logger:
        # Mock generator 抛出异常
        mock_generator = AsyncMock()
        mock_generator.__aiter__ = lambda self: self
        mock_generator.__anext__ = AsyncMock(side_effect=RuntimeError("Test error"))
        mock_generator.aclose = AsyncMock()

        # Mock instance
        mock_inst = MagicMock()
        mock_inst.async_stream_infer = MagicMock(return_value=mock_generator)
        mock_inst.async_cancel = AsyncMock()
        mock_inst.async_end = AsyncMock()
        mock_inst._active = asyncio.Event()
        mock_inst._active.set()

        # Mock engine
        with patch('lmdeploy.serve.async_engine.AsyncEngine.__init__', lambda self, *args, **kwargs: None):
            engine = AsyncEngine.__new__(AsyncEngine)
            engine.id2step = {test_session_id: 0}
            engine.backend = 'pytorch'
            engine.tokenizer = MagicMock()
            engine.tokenizer.encode = MagicMock(return_value=[1, 2, 3])
            engine.tokenizer.eos_token_id = 2
            engine.stop_words = []
            engine.session_len = 2048
            engine.request_logger = MagicMock()
            engine.request_logger.log_inputs = MagicMock()
            engine.chat_template = MagicMock()
            engine.chat_template.messages2prompt = MagicMock(return_value="test")
            engine.hf_gen_cfg = {}

            @asynccontextmanager
            async def mock_model_inst(session_id, traceid=None):
                yield mock_inst
            engine.model_inst = mock_model_inst

            async def mock_get_prompt_input(*args, **kwargs):
                return {'prompt': 'test', 'input_ids': [1, 2, 3]}
            engine._get_prompt_input = mock_get_prompt_input

            # 测试 generate() 并收集错误响应
            error_response = None
            async for gen_out in engine.generate(
                messages="test",
                session_id=test_session_id,
                traceid=test_traceid,
                stream_response=True
            ):
                print(gen_out)
                if gen_out.finish_reason == 'error':
                    error_response = gen_out
                    break

            # 验证错误响应
            assert error_response is not None
            assert error_response.finish_reason == 'error'
            assert test_traceid in error_response.response
            assert error_response.generate_token_len == 0

            # 验证日志包含 traceid
            error_calls = [call for call in mock_logger.error.call_args_list
                          if call and 'traceid' in str(call)]
            assert len(error_calls) > 0
            assert test_traceid in str(error_calls[0])
