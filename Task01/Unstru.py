from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json

file_path = "./uploads"
base_file_name = "India-Leave-Policy"

def main():
    elements = partition_pdf(filename=f"{file_path}/{base_file_name}.pdf" , include_page_breaks = True , strategy="hi_res")
    print(type(elements))
    for x in elements:
        print(x)

if __name__ == "__main__":
    main()