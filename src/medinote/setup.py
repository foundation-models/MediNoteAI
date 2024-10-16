from setuptools import setup, find_packages

setup(
    name='medinoteai',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "torch",
        "fastchat",
        "fastapi",
        "openai",
        "pandarallel",
        "llama_index",
        "weaviate-client"
    ],
    author='Hossein Akhlaghpour',
    author_email='hossein.objectj@gmail.com',
    description='MediNote AI transforms medical note-taking with AI, streamlining documentation for efficient, '
                'accurate patient records, and freeing up time for direct patient care. ',
    long_description=open('readme.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/foundation-models/MediNoteAI',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
