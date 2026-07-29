import glob

from bs4 import BeautifulSoup
import pandas as pd
import re


def process_html_without_section_names(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    # 移除 #pagetop锚点相关的部分
    cleaned_html = re.sub(r'<A NAME="pagetop"></A>\s*<HR[^>]*>', '', html_content, flags=re.IGNORECASE)


    # 重新解析 HTML
    soup = BeautifulSoup(cleaned_html, "html.parser")

    # 提取所有段落，并清理换行符
    paragraphs = soup.find_all("p")
    paragraph_texts = [
        p.get_text(separator=' ', strip=True).replace('\n', ' ').replace('[Return to top]', '').strip()
        for p in paragraphs if p.get_text(strip=True)
    ]
    # [Return to top]


    # 创建 DataFrame，每段作为一个 cell
    paragraph_df = pd.DataFrame(paragraph_texts, columns=["details"])

    # 保存为 Excel 文件
    output_file = file_path.replace(".htm", ".xlsx")
    output_file = './reformted_html_files/before_2009/' + output_file.split('/')[-1]
    paragraph_df.to_excel(output_file, index=False)


def process_html_with_section_names(file_path: str):
    # Function to extract content from HTML files

    sections = []  # 存储标题（section名称）
    details = []  # 存储正文内容（段落）

    # 打开并读取HTML文件内容
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
        # content = re.sub(r'<A NAME="pagetop"></A>.*?<HR.*?>', '', content, flags=re.DOTALL)
        soup = BeautifulSoup(content, 'html.parser')

        # 当前所处的section标题
        current_section = None

        # 遍历所有的标题和段落标签
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'strong', 'p']):
            if tag.name in ['h1', 'h2', 'h3', 'h4', 'strong']:
                # 更新当前section标题
                current_section = tag.get_text(strip=True)
            elif tag.name == 'p':
                # 提取正文段落文本
                text = tag.get_text(strip=True)
                if text:  # 忽略空文本
                    sections.append(current_section if current_section else "")
                    details.append(text)
    # sections, details = extract_content_with_sections(html_path)

    # 创建DataFrame
    df = pd.DataFrame({
        "section_name": sections,
        "details": details
    })
    output_file = file_path.replace(".htm", ".xlsx")
    output_file = './reformted_html_files/after_2009/' + output_file.split('/')[-1]
    df.to_excel(output_file, index=False)

def run():
    before_2009 = glob.glob('./fomc_minutes_html/*.htm')
    after_2009 = glob.glob('./html_files/*.htm')

    print(f'Processing {len(after_2009)} files after 2009 ...')
    n = 0
    for file in after_2009:
        try:
            process_html_with_section_names(file)
            n += 1
            print(f'Processed {n} of {len(after_2009)} files ...')
        except Exception as e:
            n += 1
            print(f'Error processing {file}, Error: {e}')
            continue

    n = 0
    for file in before_2009:
        try:
            process_html_without_section_names(file)
            n += 1
            print(f'Processed {n} of {len(before_2009)} files ...')
        except Exception as e:
            print(f'Error processing {file}, Error: {e}')
            n += 1
            continue
    print('Finished processing all files ...')

if __name__ == '__main__':
    run()



