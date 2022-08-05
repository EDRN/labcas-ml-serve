import asyncio
import aiohttp
import datetime

async def get(session: aiohttp.ClientSession, **kwargs) -> dict:
    url = "http://127.0.0.1:8080/Alphan/predict?resource_name=8_1.png"
    print(f"Requesting {url}")
    start=datetime.datetime.now()
    resp = await session.request('GET', url=url, **kwargs)
    data = await resp.json()
    print("Received data ", data, datetime.datetime.now()-start)
    return data


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
        htmls = await asyncio.gather(*tasks, return_exceptions=True)
        return htmls


if __name__ == '__main__':
    asyncio.run(main())