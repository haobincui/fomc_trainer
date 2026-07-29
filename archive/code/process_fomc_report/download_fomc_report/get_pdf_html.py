import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def get_fomc_after_2020():

    # FOMC 页面地址
    # base_url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    base_url = 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'

    # 创建保存文件的目录
    html_folder = "html_files"
    pdf_folder = "pdf_files"
    os.makedirs(html_folder, exist_ok=True)
    os.makedirs(pdf_folder, exist_ok=True)

    # 下载网页内容
    response = requests.get(base_url)
    response.raise_for_status()

    # 使用 BeautifulSoup 解析 HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # 筛选链接
    html_links = []
    pdf_links = []

    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(base_url, href)

        if href.endswith('.htm') or href.endswith('.html'):
            html_links.append(full_url)
        else:
            continue
        # elif href.endswith('.pdf'):
        #     pdf_links.append(full_url)

    print(f"发现 {len(html_links)} 个 HTML 文件，{len(pdf_links)} 个 PDF 文件。")

    # 下载 HTML 文件
    for url in html_links:
        if 'fomcminutes' in url:
            try:
                filename = url.split('/')[-1]
                path = os.path.join(html_folder, filename)

                r = requests.get(url)
                r.raise_for_status()

                with open(path, 'w', encoding='utf-8') as f:
                    f.write(r.text)

                print(f"已保存 HTML：{filename}")
            except Exception as e:
                print(f"HTML 下载失败 {url}：{e}")
        else:
            print(f"跳过 {url}")
            continue
    print("全部文件下载完成。")

def get_fomc_before_2020():
    import os
    import requests
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin

    # FOMC 历史记录页面 URL
    base_url = "https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm"
    download_folder = "fomc_minutes_html"

    # 创建保存 HTML 文件的文件夹
    os.makedirs(download_folder, exist_ok=True)

    # 获取页面 HTML
    response = requests.get(base_url)
    response.raise_for_status()  # 如果请求失败，抛出异常

    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找所有年份的链接
    year_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('/monetarypolicy/fomchistorical') and href.endswith('.htm'):
            full_url = urljoin(base_url, href)
            year_links.append(full_url)

    print(f"发现 {len(year_links)} 个年份链接，正在处理...")

    # 遍历每个年份链接，查找并下载会议记录的 HTML 文件
    target_year = ["1995", "1994", "1993"]
    for year_url in year_links:
        if year_url.split('/')[-1].split('.')[0].split('fomchistorical')[1] not in target_year:
            continue
        try:
            year_response = requests.get(year_url)
            year_response.raise_for_status()

            year_soup = BeautifulSoup(year_response.text, 'html.parser')

            # 查找指向会议记录的链接
            minutes_links = []
            for link in year_soup.find_all('a', href=True):
                href = link['href']
                if 'minutes' in href and (href.endswith('.htm') or href.endswith('.html')):
                    full_url = urljoin(year_url, href)
                    minutes_links.append(full_url)
                elif 'fomc/MINUTES/' in href and href.endswith('.htm'):
                    full_url = urljoin(year_url, href)
                    minutes_links.append(full_url)

            print(f"在 {year_url} 中发现 {len(minutes_links)} 个会议记录链接，正在下载...")

            # 下载会议记录的 HTML 文件
            for url in minutes_links:
                try:
                    filename = url.split('/')[-1]
                    filepath = os.path.join(download_folder, filename)

                    file_response = requests.get(url)
                    file_response.raise_for_status()

                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(file_response.text)

                    print(f"已保存：{filename}")
                except Exception as e:
                    print(f"下载失败 {url}，原因：{e}")

        except Exception as e:
            print(f"处理年份链接 {year_url} 时出错，原因：{e}")

    print("全部下载完成。")

if __name__ == '__main__':
    # get_fomc_after_2020()
    get_fomc_before_2020()
