import os
import myModules
from atlassian import Confluence
from tqdm import tqdm
import json


os.environ['atlassianUserEmail'] = 'dantanlianxyx@gmail.com' 
os.environ['atlassianAPIToken'] = 'ATATT3xFfGF0aVaZRzbU5KjDyEJgpDmCY8_b7CkThNxBWU2xoPeduFwMgLHWlrBSixWpGWwxAGbMsos7Pll-FVr3xcDek09WqmjxYoISXDDYo5ogcdUV2dsBvIKH4SSfAiv32zVyl-RsAQmcVlHw7RfwGjrbCq4TdurGo9QcCSza1l74wLiU5Sc=F2584998'


def export_page_as_html(atlassian_site,page_id):
    user_name = os.environ["atlassianUserEmail"]
    api_token = os.environ["atlassianAPIToken"]

    # page_name = myModules.get_page_name(atlassian_site, page_id, user_name, api_token)
    page_content_response = myModules.get_body_export_view(atlassian_site, page_id, user_name, api_token)
    page_content = page_content_response.json()['body']['export_view']['value']
    page_title = page_content_response.json()['title']

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    html_of_page = myModules.dump_html(
        arg_site=atlassian_site,
        arg_html=page_content,
        arg_title=page_title,
        arg_page_id=page_id,
        # arg_outdir_base=output_dir,
        arg_outdir_content=output_dir,
        arg_page_labels='',
        arg_page_parent='',
        arg_username=user_name,
        arg_api_token=api_token,
        arg_sphinx_compatible=False,
        arg_sphinx_tags=False,
        arg_html_output=True,
        arg_rst_output=False,
        arg_show_labels=False
    )
    return html_of_page


# def download_all_confluence_page(output_folder):
#     atlassian_site = 'singularity-systems'
#     confluence = Confluence(
#         url='https://singularity-systems.atlassian.net',
#         username='dantanlianxyx@gmail.com',
#         password='ATATT3xFfGF0aVaZRzbU5KjDyEJgpDmCY8_b7CkThNxBWU2xoPeduFwMgLHWlrBSixWpGWwxAGbMsos7Pll-FVr3xcDek09WqmjxYoISXDDYo5ogcdUV2dsBvIKH4SSfAiv32zVyl-RsAQmcVlHw7RfwGjrbCq4TdurGo9QcCSza1l74wLiU5Sc=F2584998',
#         api_version='cloud',
#         cloud=True)

#     spaces = confluence.get_all_spaces(start=0, limit=500, expand=None)['results']
#     list_of_spaces = []
#     for i in spaces:
#         if i['type'] != 'personal' and i['key'] != 'JohnnyDeci':
#             list_of_spaces.append(i['key'])

#     list_of_pages = []
#     for space in tqdm(list_of_spaces):
#         try:
#             progress_bar = tqdm(ncols=100, dynamic_ncols=True)
#             print(space)
#             count = 0
#             while True:
#                 pages = confluence.get_all_pages_from_space(space= space, start=count, limit=100)
#                 list_of_pages += pages
#                 progress_bar.update(1)
#                 if len(pages) < 100:
#                     break
#                 count += 100
#                 progress_bar.close()
#         except Exception as Argument:
#             print('报错：', Argument)
        
#     print(f'Total number of pages is {len(list_of_pages)}')
#     log_file = os.path.join(output_folder, 'download_log.json')

#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     if os.path.exists(log_file):
#         with open(log_file, 'r') as f:
#             download_log = json.load(f)
#     else:
#         download_log = {}

#     for page in tqdm(list_of_pages):
#         page_id = page['id']
#         title = page['title'].replace('.','_').replace('/','_')
#         version = confluence.history(page_id)['lastUpdated']['when'].replace('.','_')
#         file_name = f'id_{page_id}_title_{title}.html'
#         output_file = os.path.join(output_folder, file_name)

#         if page_id not in download_log or download_log[page_id] != version:
#             html = export_page_as_html(atlassian_site=atlassian_site,page_id=page_id)
#             with open(output_file,  'w', encoding='utf-8') as html_file:
#                 html_file.write(html)
            
#             download_log[page_id] = version
#         else:
#             print(f'file {page_id}_{title}_{version} already downloaded')


#         with open(log_file, 'w') as f:
#             json.dump(download_log, f)

#     return output_folder

# if __name__ == "__main__":
#     out_dic = '/data2/yixu/confluence_html_pages'
#     download_all_confluence_page(out_dic)