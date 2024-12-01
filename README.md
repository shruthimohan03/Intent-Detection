# INTENT DETECTION  
---

## 1. Problem Framing  

### a. Framing the Problem  
This is an **ML multi-classification problem**, where the goal is to classify each text input into one of the predefined intent categories accurately. The task involves learning patterns between the input text and intent labels using machine learning or deep learning models.  

### b. Approaches and Considerations  
- **Initial Approach**:  
  I started by considering **BERT embeddings** combined with a **Random Forest model**. This approach aimed to utilize the rich contextual information provided by BERT while leveraging the simplicity of traditional ML models. However, this approach yielded an accuracy of only **55%**, likely due to the small dataset (~300 rows).  

- **Deep Learning Architecture**:  
  To improve performance, I shifted to a **simple neural network with dropout layers** for regularization. This yielded a better accuracy of **65%**.

- **Addressing Class Imbalance**:  
  I experimented with **class weights** to handle imbalanced data, which improved accuracy to **68%**. However, the improvement was modest.

- **Hyperparameter Tuning**:  
  Finally, I focused on optimizing the model through **GridSearchCV**. By fine-tuning parameters like dropout rate, batch size, epochs, and dense layer units, the model achieved an accuracy of **~70%**, marking a meaningful improvement.

---

## 2. Model Building  

### Workflow  

#### **Notebook 1: Random Forest Model**  
1. **Cleaning**: Checked for missing values; none were found.  
2. **Preprocessing**: Performed tokenization and stop-word removal.  
3. **Data Splitting**: Divided the dataset into training and testing sets.  
4. **Feature Extraction**: Generated **BERT embeddings** for text data.  
5. **Model Training**: Trained a Random Forest classifier.  
6. **Result**: Achieved **55% accuracy**.  

#### **Notebook 2: Neural Network + Class Weights + Hyperparameter Tuning**  
1. **Cleaning**: Verified no missing values.  
2. **Preprocessing**: Tokenized text and removed stop words.  
3. **Label Encoding**: Encoded labels into numerical format for compatibility with neural networks.  
4. **Data Splitting**: Split the dataset into training and testing sets.  
5. **Feature Extraction**: Generated **BERT embeddings** for the text data.  
6. **Model Training**: Trained a simple neural network with dropout layers.  
   - **Initial Result**: Achieved **65% accuracy**.  
7. **Class Weights**: Applied class weights to address imbalances, improving accuracy to **68%**.  
8. **Hyperparameter Tuning**: Conducted parameter optimization using **GridSearchCV**:  
   - Tweaked parameters such as dropout rate, epochs, and batch size.  
   - **Final Result**: Achieved **~70% accuracy**.  

---

## 3. Analysis  

### a. Justification for Results  
- **Incremental Improvements**:  
  - The accuracy improved from **55%** (Random Forest) to **65%** (Neural Network) and then to **70%** after class weights and hyperparameter tuning.  
- **Contextual Embeddings**:  
  - Leveraging **BERT embeddings** enabled the model to capture rich semantic and contextual features, crucial for intent detection.  
- **Tuning Effectiveness**:  
  - The use of **GridSearchCV** demonstrated that systematic parameter tuning significantly enhanced the model's performance.  

### b. Suggestions for Improvement  
1. **Dataset Augmentation**:  
   - Increase the dataset size or use **data augmentation** techniques to provide the model with more diverse examples, reducing overfitting.  

2. **Advanced Architectures**:  
   - Experiment with advanced transformer models like **RoBERTa** or **DistilBERT**, which often outperform BERT in downstream tasks.  

3. **Attention Mechanisms**:  
   - Introduce **attention layers** to focus on the most critical parts of the input text, improving classification performance.  

4. **Ensemble Learning**:  
   - Combine the strengths of **Random Forest** and **Neural Network** models using ensemble techniques to enhance robustness.  

---

## 4. Results Summary  

| **Approach**                  | **Accuracy** |  
|-------------------------------|--------------|  
| Random Forest + BERT Embeddings | 55%          |  
| Neural Network                | 65%          |  
| Neural Network + Class Weights | 68%          |  
| Neural Network + Tuning       | ~70%         |  

---

## 5. Tools and Libraries  

- **Python**: Language for implementation.  
- **TensorFlow/Keras**: For building neural network models.  
- **scikit-learn**: For data preprocessing, Random Forest, and GridSearchCV.  
- **Transformers (Hugging Face)**: For BERT embeddings.  

---
