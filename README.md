# Automated FAQ Generation System for University Administrative Content

This project is an analytical framework leveraging BERT-based Natural Language Processing (NLP) techniques to automate the detection of key administrative details and the generation of coherent Frequently Asked Questions (FAQs) for Universitas Negeri Jakarta (UNJ). The system aims to enhance information accessibility within UNJ’s digital ecosystem by processing unstructured, bilingual administrative content.

## 1. Project Overview

The rapid growth of online education platforms has increased the demand for efficient information retrieval systems to address repetitive administrative queries at universities. Universitas Negeri Jakarta (UNJ) faces challenges in manually creating FAQs from diverse unstructured administrative content, such as admission webpages, news announcements, and policy documents. This manual process often leads to delays in supporting students’ queries regarding registration deadlines, tuition fees, and admission requirements.

This research aims to establish a comprehensive analytical framework that utilizes advanced NLP techniques, specifically BERT-based models, to automate:
- The **detection** of key administrative details from unstructured texts.
- The **generation** of coherent and relevant FAQs.

The goal is to improve information accessibility and reduce administrative workload at UNJ, with a model that could be adaptable to other educational institutions.

## 2. Problem Statement

-   **Inefficient Manual FAQ Creation:** Manually sifting through unstructured administrative content (webpages, news, policy documents) to create and update FAQs is time-consuming and labor-intensive for UNJ staff.
-   **Delayed Student Support:** The inefficiency in FAQ management can lead to delays in providing students with timely answers to common administrative queries (e.g., registration deadlines, tuition fees, admission requirements).
-   **Information Accessibility Gap:** Students may struggle to find necessary information scattered across various unstructured sources, impacting their ability to navigate university procedures effectively.
-   **Bilingual Content Challenge:** Administrative information at UNJ is often available in both Indonesian and English, requiring a system capable of processing and generating FAQs in a bilingual context.

## 3. Proposed Solution & Methodology

The core of this research is to develop an analytical framework that involves the following key stages:

### 3.1. Data Collection and Preparation
-   **Dataset Acquisition:** Collect a bilingual (Indonesian-English) dataset comprising 500–1,000 text samples.
    -   **Sources:** UNJ’s official website, Learning Management System (LMS), social media channels, admission webpages, news announcements, and policy documents.
    -   **Methods:** Web scraping techniques and manual extraction.
-   **Data Preprocessing:** Clean and prepare the collected text data for NLP model training. This may include text normalization, tokenization, and handling of bilingual content.

### 3.2. NLP Model Development
-   **Model Selection:** Utilize a multilingual Bidirectional Encoder Representations from Transformers (BERT) model as the foundation.
-   **Fine-Tuning:** Fine-tune the pre-trained multilingual BERT model on the collected UNJ-specific dataset. This will adapt the model to understand the nuances of university administrative language and topics in both Indonesian and English.

### 3.3. FAQ Detection and Generation Framework
-   **Key Information Detection:** Implement mechanisms within the fine-tuned BERT model to identify and extract key administrative details from unstructured text.
    -   *Example:* Detecting "registration deadline: June 1st" or "tuition fee: IDR 5,000,000".
-   **Question-Answer Pair Generation:** Develop a component that uses the detected information to generate coherent and contextually relevant question-answer pairs (FAQs).
    -   *Example:* From "registration deadline: June 1st", generate "Q: When is the university registration deadline? A: The registration deadline is June 1st."
-   **Coherence and Relevance:** Ensure generated FAQs are natural-sounding, accurate, and directly address potential student queries.

### 3.4. System Integration (Conceptual)
-   The framework is designed to be a backend system that can potentially feed into UNJ's existing student support portals or a dedicated FAQ interface.

## 4. Conceptual Usage (Framework Workflow)

The system, once developed, would operate as follows:

1.  **Input:** Unstructured administrative text documents (Indonesian/English).
    ```python
    # Example pseudo-code for using the framework
    # from unj_faq_generator import FAQFramework

    # framework = FAQFramework(model_path="path/to/fine-tuned-bert")
    # raw_text_document = "Pengumuman penting: Pendaftaran mahasiswa baru dibuka hingga 1 Juni. Biaya kuliah semester ini adalah Rp 5.000.000."

    # generated_faqs = framework.process_document(raw_text_document)
    # for faq in generated_faqs:
    # print(f"Q: {faq.question}")
    # print(f"A: {faq.answer}\n")
    ```

2.  **Processing:**
    -   The input text is fed into the fine-tuned multilingual BERT model.
    -   The model performs **key information detection** (e.g., identifies dates, amounts, keywords like "pendaftaran", "biaya kuliah").
    -   The **FAQ generation module** uses these detected entities to formulate relevant questions and extract/generate corresponding answers.

3.  **Output:** A set of structured FAQ pairs.
    ```
    # Example conceptual output
    Q: Kapan batas waktu pendaftaran mahasiswa baru?
    A: Pendaftaran mahasiswa baru dibuka hingga 1 Juni.

    Q: Berapa biaya kuliah semester ini?
    A: Biaya kuliah semester ini adalah Rp 5.000.000.
    ```

## 5. Evaluation Metrics

The performance of the FAQ generation system will be evaluated using standard NLP metrics, including:
-   **BLEU (Bilingual Evaluation Understudy):** Measures the similarity of the generated text to reference texts, focusing on n-gram precision.
-   **ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence):** Evaluates the quality of summaries/generated text by comparing the longest common subsequence with reference texts.
-   **METEOR (Metric for Evaluation of Translation with Explicit ORdering):** Considers synonymy and stemming, providing a more nuanced evaluation than BLEU.
-   **F1-Score:** The harmonic mean of precision and recall, often used for evaluating the accuracy of the information detection component.

## 6. Expected Outcomes and Contributions

-   **Scalable FAQ System:** A robust and scalable system capable of processing large volumes of unstructured administrative content to generate FAQs automatically.
-   **Multilingual Support:** Effective FAQ generation for both Indonesian and English content, catering to UNJ's bilingual environment.
-   **Reduced Administrative Workload:** Automation of FAQ creation will significantly lessen the manual effort required from administrative staff.
-   **Improved Student Support:** Faster access to accurate information for students, leading to enhanced satisfaction and a smoother university experience.
-   **Adaptable Model:** The developed framework and fine-tuned model can serve as a blueprint for other educational institutions facing similar challenges.
-   **Contribution to NLP in Education:** This study will contribute to the field of NLP applications in educational administration, specifically addressing the gap in automated FAQ generation for unstructured, bilingual university content.

## 7. Note

-   **Dataset Quality:** The success of the fine-tuned BERT model heavily depends on the quality, quantity, and representativeness of the collected bilingual dataset.
-   **Domain Specificity:** The model will be specialized for UNJ's administrative domain. Generalization to other domains might require further fine-tuning.
-   **Ethical Considerations:** Ensuring data privacy and avoiding biases in generated FAQs will be important.
-   **Iterative Development:** The framework will likely require iterative refinement and evaluation to achieve optimal performance.

## 8. Academic Disclaimer

This project is conceptualized as a research initiative. While it aims to develop a functional framework, further development, rigorous testing, and integration efforts would be necessary for deployment in a live production environment at Universitas Negeri Jakarta.

## 9. Technologies & Acknowledgments (Potential)

-   **Primary Technology:** BERT (Bidirectional Encoder Representations from Transformers)
-   **Key Libraries/Frameworks:**
    -   [Hugging Face Transformers](https://huggingface.co/transformers/)
    -   [PyTorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org/)
    -   [Scikit-learn](https://scikit-learn.org/)
    -   [NLTK](https://www.nltk.org/) / [spaCy](https://spacy.io/) (for text preprocessing)
-   **Inspiration:** Research in automated question generation, information extraction, and NLP applications in the education sector.
