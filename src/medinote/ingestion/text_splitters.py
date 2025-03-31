from apps.knowledge.models import KnowledgeRepository, Knowledge, KnowledgeContent


def main_and_subtitle_splitter(text, main_titles, subtitles):
    lines = text.split('\n')
    sections = []
    current_main_title = None
    current_subtitle = None
    content_accumulator = []  # To accumulate content for each section

    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        # Check if the line is a main title or subtitle
        if line in main_titles:
            # Save the previous section before starting a new main title
            if current_main_title:
                sections.append({
                    "main_title": current_main_title,
                    "subtitle": current_subtitle,
                    "content": ' '.join(content_accumulator).strip()
                })
                content_accumulator = []

            current_main_title = line
            current_subtitle = None
        elif line in subtitles and current_main_title:
            # Save the previous subtitle section before starting a new subtitle
            if current_subtitle:
                sections.append({
                    "main_title": current_main_title,
                    "subtitle": current_subtitle,
                    "content": ' '.join(content_accumulator).strip()
                })
                content_accumulator = []

            current_subtitle = line
        else:
            content_accumulator.append(line)

    # Add the last section
    if current_main_title:
        sections.append({
            "main_title": current_main_title,
            "subtitle": current_subtitle,
            "content": ' '.join(content_accumulator).strip()
        })

    return sections


def fixed_size_splitter(text):
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=512,
        chunk_overlap=20
    )
    docs = text_splitter.create_documents([text])
    return docs


def sentence_splitter(text):
    docs = text.split(".")
    return docs


def nltk_splitter(text):
    from langchain.text_splitter import NLTKTextSplitter
    text_splitter = NLTKTextSplitter(
        separator="\n\n",
        chunk_size=512,
        chunk_overlap=20
    )
    docs = text_splitter.split_text(text)
    return docs


def recursive_splitter(text):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        separator="\n\n",
        chunk_size=512,
        chunk_overlap=20
    )
    docs = text_splitter.create_documents([text])
    return docs


def markdown_splitter(markdown_text):
    from langchain.text_splitter import MarkdownTextSplitter
    markdown_splitter = MarkdownTextSplitter(chunk_size=200, chunk_overlap=0)
    docs = markdown_splitter.create_documents([markdown_text])
    return docs


def latex_splitter(latex_text):
    from langchain.text_splitter import LatexTextSplitter
    latex_splitter = LatexTextSplitter(chunk_size=100, chunk_overlap=0)
    docs = latex_splitter.create_documents([latex_text])
    return docs


def prepare_repo(repo: KnowledgeRepository):
    knowledge = repo.knowledge

    handbook_file = open(repo.file.path)

    # Read main titles from file
    with open(repo.main_titles_file.path, "r") as file:
        main_titles = [line.strip() for line in file.readlines()]

    # Read subtitles from file
    with open(repo.sub_titles_file.path, "r") as file:
        subtitles = [line.strip() for line in file.readlines()]

    # Read the handbook text
    with open(repo.file.path, "r") as file:
        handbook_text = file.read()

    # Process the handbook text
    sections = main_and_subtitle_splitter(handbook_text, main_titles, subtitles)

    print(len(main_titles))
    print(len(subtitles))

    extracted_main_titles = []
    extracted_sub_titles = []

    KnowledgeContent.objects.filter(knowledge_repo=repo).delete()

    # Output the sections
    for section in sections:
        extracted_main_titles.append(section['main_title'])
        extracted_sub_titles.append(section['subtitle'])
        main_title = section['main_title']
        sub_title = section['subtitle']
        text = section['content']

        title = ""
        if str(sub_title).lower() == "none":
            title = main_title
            titles = main_title
        else:
            title = f"{main_title} - {sub_title}"
            titles = f"{main_title}, {sub_title}"

        knowledge_content = KnowledgeContent.objects.create(
            knowledge_repo=repo,
            knowledge=knowledge,
            title=title,
            text=text,
            titles=titles
        )

        print(f"Main Title: {section['main_title']}")
        print(f"Subtitle: {section['subtitle']}")
        # print(f"Content:\n{section['content']}")
        # print("--------------------------------")

    print(len(list(set(extracted_main_titles))))
    print(len(extracted_sub_titles))
