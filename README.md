# 🔬 Polyp Segmentation Project 

This repository contains the research and results from our project on polyp segmentation using deep learning. Our goal was to evaluate and improve existing segmentation methods for polyp detection, focusing on generalization across multiple small benchmark datasets. 

## 📄 Report.pdf

The main document for this project is `Report.pdf`. It details our approach, methodologies, experiments, and results. Here's a quick overview:

-   **Title**: Polyp Segmentation : CS 7643
-   **Authors**: Ashok Kamath, Zahin Awosaf, Quincy Nickens
-   **Institution**: Georgia Institute of Technology

### 📖 Abstract

We explored various deep learning architectures for polyp segmentation, including U-Net, U-Net++, ResUNet++, and DUCK-Net, along with our custom model, HybridNet. Our focus was on improving generalization across small medical imaging datasets. We used standard preprocessing, data augmentation, and architectural modifications. Results show that while established models perform well individually, customized models like ResUNet++ achieved stronger generalization.

### 📋 Table of Contents

1.  **Introduction/Background/Motivation**
    -   Importance of early polyp detection for colorectal cancer prevention.
    -   Focus on small polyp datasets for cost-effectiveness.
    -   Challenge of generalization across different datasets.
2.  **Architectures**
    -   Detailed descriptions of U-Net, U-Net++, ResUNet++, DUCK-Net, and HybridNet.
3.  **Approach**
    -   Training and testing methodologies.
    -   Data preprocessing and augmentation.
4.  **Experiments and Results**
    -   Evaluation metrics: Intersection over Union (IoU) and Dice score.
    -   Results for each model trained on individual and combined datasets.
5.  **Conclusion**
    -   Summary of findings.
    -   Importance of model simplicity, diverse training data, and targeted augmentations.
6.  **Illustrations, graphs, and photographs**
    -   Various tables summarizing test results and architectural diagrams.
7.  **Work Division**
    -   Contributions of each team member.
8.  **References**
    -   Citations for all the sources used in the report.

### 📊 Key Findings

-   ResUNet++ achieved the highest overall performance with strong generalization across datasets.
-   Data augmentation significantly improved model performance, especially for ResUNet++.
-   DUCK-Net performed better with a pruned architecture, suggesting model complexity can hinder performance on small datasets.
-   HybridNet provided a competitive alternative to established networks.

### 🚀 Getting Started

To understand the project in depth, please refer to `Report.pdf`. The report contains all the details about our methodology, experiments, and results.

### 🤝 Contributors

-   Ashok Kamath (ashok.kamath19@gmail.com)
-   Zahin Awosaf
-   Quincy Nickens

### 🔗 Links

-   GitHub Repository: [Link to your GitHub repository] (Replace with your repo link)
-   Report PDF: [Google Drive Link to Report.pdf](https://drive.google.com/open?id=1VzUI0Kz1XhvWM8ACgkhGGKlk9FUNsWtt)

### 📅 Date

Tuesday, 13 May 2025
