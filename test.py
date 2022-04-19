import asyncio
import aiohttp

async def get(session: aiohttp.ClientSession, **kwargs):
    url = "http://127.0.0.1:8080/alphan/predict?resource_name=8_1.png"
    print(f"Requesting {url}")
    resp = await session.request('GET', url=url, **kwargs)
    # data = await resp.json()
    # error = await resp.reason
    # print("Received data ", data, 'error:', error)
    return resp


async def main(**kwargs):
    # Asynchronous context manager.  Prefer this rather
    # than using a different session for each GET request
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(5):
            tasks.append(get(session=session, **kwargs))
        # asyncio.gather() will wait on the entire task set to be
        # completed.  If you want to process results greedily as they come in,
        # loop over asyncio.as_completed()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        return responses


if __name__ == '__main__':
    import datetime
    start=datetime.datetime.now()
    responses=asyncio.run(main())
    for resp in responses:
        print('status:', resp.status)
    print('time taken:', datetime.datetime.now()- start)