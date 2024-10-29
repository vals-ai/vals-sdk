import asyncio

from vals.sdk.v2.util import get_ariadne_client


# TODO: These two methods may be part of a Run Object
async def run_status(run_id: str) -> str:
    client = get_ariadne_client()
    result = await client.run_status(run_id=run_id)
    return result.run.status


async def wait_for_run_completion(run_id: str) -> str:
    """
    Block a process until a given run has finished running.

    """
    await asyncio.sleep(1)
    status = "in_progress"
    while status == "in_progress":
        status = await run_status(run_id)
        await asyncio.sleep(1)

    return status
