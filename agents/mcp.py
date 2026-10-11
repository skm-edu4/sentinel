import asyncio
import atexit
import json
import sys
import threading
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_TIMEOUT = 20.0
SHUTDOWN_TIMEOUT = 5.0


class McpCallError(RuntimeError):
    pass


class _ServerActor:
    def __init__(self, script: str):
        self.script = script
        self.params = StdioServerParameters(
            command=sys.executable,
            args=[str(BASE_DIR / "servers" / script)],
            cwd=str(BASE_DIR),
        )
        self.queue: asyncio.Queue = asyncio.Queue()
        self.task: asyncio.Task | None = None

    async def _connect(self):
        stream_cm = stdio_client(self.params)
        read, write = await stream_cm.__aenter__()
        session_cm = ClientSession(read, write)
        session = await session_cm.__aenter__()
        try:
            await session.initialize()
        except BaseException:
            await session_cm.__aexit__(*sys.exc_info())
            await stream_cm.__aexit__(*sys.exc_info())
            raise

        async def close():
            for cm in (session_cm, stream_cm):
                try:
                    await cm.__aexit__(None, None, None)
                except Exception:
                    pass

        return session, close

    async def run(self):
        session = None
        close = None
        while True:
            item = await self.queue.get()
            if item is None:
                break
            future, tool, args = item
            try:
                if session is None:
                    session, close = await self._connect()
                result = await session.call_tool(tool, args)
                self._deliver(future, self._unwrap(result))
            except McpCallError as exc:
                self._deliver(future, exc)
            except Exception as exc:
                self._deliver(future, exc)
                if close is not None:
                    await close()
                session = None
                close = None
        if close is not None:
            await close()

    @staticmethod
    def _deliver(future: asyncio.Future, value) -> None:
        if future.done():
            return
        if isinstance(value, BaseException):
            future.set_exception(value)
        else:
            future.set_result(value)

    @staticmethod
    def _unwrap(result) -> object:
        if result.is_error:
            messages = [c.text for c in result.content if hasattr(c, "text")]
            raise McpCallError(" ".join(messages))
        structured = result.structured_content
        if isinstance(structured, dict):
            if set(structured) == {"result"}:
                return structured["result"]
            return structured
        text = result.content[0].text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text


_LOOP: asyncio.AbstractEventLoop | None = None
_THREAD: threading.Thread | None = None
_ACTORS: dict[str, _ServerActor] = {}
_LOCK = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _LOOP, _THREAD
    with _LOCK:
        if _LOOP is None:
            _LOOP = asyncio.new_event_loop()
            _THREAD = threading.Thread(
                target=_LOOP.run_forever, name="mcp-session-loop", daemon=True
            )
            _THREAD.start()
            atexit.register(_shutdown)
    return _LOOP


async def _ensure_actor(script: str) -> _ServerActor:
    actor = _ACTORS.get(script)
    if actor is None:
        actor = _ServerActor(script)
        actor.task = asyncio.create_task(actor.run(), name=f"mcp-actor-{script}")
        _ACTORS[script] = actor
    return actor


async def _invoke(script: str, tool: str, args: dict, timeout: float) -> object:
    actor = await _ensure_actor(script)
    future = asyncio.get_running_loop().create_future()
    await actor.queue.put((future, tool, args))
    return await asyncio.wait_for(asyncio.shield(future), timeout)


def _submit(script: str, tool: str, args: dict, timeout: float) -> object:
    loop = _ensure_loop()
    try:
        return asyncio.run_coroutine_threadsafe(
            _invoke(script, tool, args, timeout), loop
        ).result(timeout + 5.0)
    except McpCallError:
        raise
    except Exception as exc:
        raise exc


def call_tool(
    server_script: str, tool: str, args: dict, timeout: float = DEFAULT_TIMEOUT
) -> object:
    try:
        return _submit(server_script, tool, args, timeout)
    except McpCallError:
        raise
    except (TimeoutError, asyncio.TimeoutError) as exc:
        raise McpCallError(f"{server_script}.{tool}: timed out after {timeout}s") from exc
    except Exception:
        try:
            return _submit(server_script, tool, args, timeout)
        except McpCallError:
            raise
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise McpCallError(
                f"{server_script}.{tool}: timed out after {timeout}s"
            ) from exc
        except Exception as exc:
            raise McpCallError(f"{server_script}.{tool}: {exc}") from exc


def _shutdown() -> None:
    loop = _LOOP
    if loop is None:
        return

    async def _close_all():
        actors = list(_ACTORS.values())
        for actor in actors:
            actor.queue.put_nowait(None)
        for actor in actors:
            try:
                await asyncio.wait_for(actor.task, SHUTDOWN_TIMEOUT)
            except Exception:
                pass
        _ACTORS.clear()

    try:
        asyncio.run_coroutine_threadsafe(_close_all(), loop).result(
            SHUTDOWN_TIMEOUT + 1.0
        )
    except Exception:
        pass
    loop.call_soon_threadsafe(loop.stop)
    if _THREAD is not None:
        _THREAD.join(timeout=2.0)
