import asyncio
import time

class AsyncRuntime:
    def __init__(self, max_concurrent=4):
        self.max_concurrent = max_concurrent

    async def _run_agent(self, agent, task):
        start = time.time()
        try:
            result = await asyncio.to_thread(agent.act, task)
            status = "success"
        except Exception as e:
            result = {"error":str(e)}
            status = "error"
        end = time.time()

        return {
            "agent":agent.__class__.__name__,
            "task":task,
            "result":result,
            "status":status,
            "duration_ms":(end-start) * 1000
        }
            

    async def run_batch(self, assignments, get_agent_fn):
        sem = asyncio.Semaphore(self.max_concurrent)
        tasks = []

        async def run_with_semaphore(assignment):
            async with sem:
                agent = get_agent_fn(assignment["agent"])
                return await self._run_agent(agent, assignment["task"])
        
        for a in assignments:
            tasks.append(run_with_semaphore(a))
        results = await asyncio.gather(*tasks)
        return results