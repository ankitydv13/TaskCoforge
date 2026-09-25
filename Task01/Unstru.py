from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json

file_path = "./uploads"
base_file_name = "India-Leave-Policy"

def main():
    elements = partition_pdf(
        filename=f"{file_path}/{base_file_name}.pdf" , 
        strategy="hi_res",
        infer_table_structure = True,
        include_page_breaks = True
    )
    print(type(elements[0]))
    # for element in elements:
    #     # print(f"Element Id : {element._element_id} \t Element_Type-->  {type(element).__name__} \t Element Text --> {element.text} ")
    #     # print("\n")
    #     print(element.to_dict())
        
    

if __name__ == "__main__":
    main()