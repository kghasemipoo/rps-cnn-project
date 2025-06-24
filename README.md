# 🧠 Rock–Paper–Scissors Image Classifier with CNNs

This project applies Convolutional Neural Networks (CNNs) to classify hand gesture images from the classic Rock–Paper–Scissors game. The objective was to design and train a model using **sound machine learning methodology**, rather than purely optimizing for accuracy or speed.

---

## 🎯 Project Objective

The goal of the project was to build a CNN-based image classifier for three hand gesture classes: **rock**, **paper**, and **scissors**, using the dataset from Kaggle.

Instead of maximizing raw performance, the project focused on:

- ✅ Clean and leak-free data preprocessing  
- ✅ Designing and comparing multiple CNN architectures  
- ✅ Applying **automatic hyperparameter tuning** (Keras Tuner)  
- ✅ Using techniques like **early stopping** and **dropout**  
- ✅ Analyzing overfitting and underfitting behaviors  
- ✅ Reporting test metrics and performance trade-offs

---

## 🧪 Dataset

- **Source**: [Kaggle Rock–Paper–Scissors Dataset](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors)
- ~2,200 labeled images (rock, paper, scissors)
- Stratified 80/10/10 split into train/val/test sets
- Resized to 150×150 pixels and normalized to [0, 1]

---

## 🧠 Model Results

| Model          | Train Time (CPU) | Test Accuracy |
|----------------|------------------|---------------|
| TinyNet        | ~2 min           | 54.6%         |
| TinyNet++      | ~9 min           | 66.4%         |
| MediumNet-fast | ~12 min          | **96.4%**     |
| MediumNet-full | ~45 min          | 95.0%         |

The best model, **MediumNet-fast**, was trained with a learning rate found using **Keras Tuner’s RandomSearch**, and used **early stopping** to avoid overfitting. It achieved high accuracy while remaining computationally efficient.

---

## 📈 Key Takeaways

- ✅ Final model reached **96.4% accuracy** in under 13 minutes of CPU training  
- ✅ Proper tuning and regularization were more effective than brute-force depth  
- ✅ The clean training pipeline and methodological rigor followed academic best practices

---

## 👨‍🎓 Author & Credits

**Project by**: Kasra Ghasemipoo  
**Instructor**: Prof. Nicolò Cesa-Bianchi  
**University**: Università degli Studi di Milano  
**Academic Year**: 2024/25  

📧 Contact:  
- kghasemipoo@gmail.com  
- kasra.ghasemipoo@studenti.unimi.it

---
