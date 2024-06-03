import asyncio
import subprocess
import warnings

import aiohttp
import async_timeout
import requests

from loguru import logger
from urllib3.exceptions import InsecureRequestWarning

headers = {
    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 "
                  "Safari/537.36",
    'accept-language': "zh-CN,zh;q=0.9,en-CN;q=0.8,en;q=0.7"
}

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

proxy = "http://localhost:12345"


async def request_one_url(session, url):
    try:
        async with async_timeout.timeout(15):
            response = await session.get(url, headers=headers, proxy=proxy)
            html = await response.text(errors="ignore")
            return html
    except Exception as aiohttp_exception:
        # logger.warning(f"aiohttp fetch {url} error: {aiohttp_exception}")
        pass

    # try:
    #     response = requests.get(url, headers=headers, verify=False, timeout=5, proxies={"http": proxy, "https": proxy})
    #     response.encoding = response.apparent_encoding
    #     html = response.text
    #     return html
    # except Exception as request_exception:
    #     logger.warning(f"requests fetch {url} error: {request_exception}")
    
    cmd = f"curl --insecure --connect-timeout 10 {url}"
    proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    # stdout, stderr = await proc.communicate()
    
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        stdout, stderr = "", "timeout"
    
    if proc.returncode != 0:
        logger.warning(f"curl {url} return {proc.returncode}")
    
    # if stderr:
    #     print(f'[stderr]\n{stderr.decode(encoding="UTF-8", errors="ignore")}')
    
    if stdout:
        html = stdout.decode(encoding="UTF-8", errors="ignore")
    else:
        html = ""
    
    # process = subprocess.Popen(["curl", "--insecure", "--connect-timeout", "10", url], stdout=subprocess.PIPE,
    #                            stderr=subprocess.DEVNULL, text=True)
    # html, error = process.communicate()

    # if error is not None:
    #     logger.warning(f"unable to fetch {url}. curl error: {error}")

    # if html is None:
    #     html = ""

    return html

html_texts = []
async def main(urls):
    global html_texts
    async with aiohttp.ClientSession() as session:
        tasks = [request_one_url(session, url) for url in urls]
        html_texts = await asyncio.gather(*tasks)

def fetch(urls):
    # asyncio.run() 通常是为单进程的事件循环设计的
    # html_texts = asyncio.run(main(urls)) 
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(urls))
    return {urls[i]: html_texts[i] for i in range(len(urls))}


