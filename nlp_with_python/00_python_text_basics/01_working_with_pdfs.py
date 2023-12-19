# region Working with PDFs
import PyPDF2

# read a pdf
myfile = open("US_Declaration.pdf", mode="rb")
pdf_reader = PyPDF2.PdfReader(myfile)
page_one = pdf_reader.pages[0]
mytext = page_one.extract_text()
myfile.close()

# copy and write to a new pdf
f = open("US_Declaration.pdf", mode="rb")
pdf_reader = PyPDF2.PdfReader(f)
first_page = pdf_reader.pages[0]
pdf_writer = PyPDF2.PdfWriter()
pdf_writer.add_page(first_page)
pdf_output = open("my_new.pdf", "wb")
pdf_writer.write(pdf_output)
pdf_output.close()
f.close()
new = open("my_new.pdf", "rb")
pdf_reader = PyPDF2.PdfReader(new)
len(pdf_reader.pages)

# grab text from pdf file
f = open("US_Declaration.pdf", "rb")
pdf_text = []
pdf_reader = PyPDF2.PdfReader(f)

for p in range(len(pdf_reader.pages)):
    page = pdf_reader.pages[p]
    pdf_text.append(page.extract_text())

for page in pdf_text:
    print(page)
    print("\n" * 5)
# endregion
