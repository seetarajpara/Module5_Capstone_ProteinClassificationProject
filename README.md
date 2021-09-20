# Module5_Capstone_ProteinClassificationProject

Deep learning project for protein classification.

## Protein Classification Problem

### Background Research
- Proteins are macromolecules responsible for all biological processes in living cells
- They are made up of amino acid chains, making up a larger sequence of these molecules
    - Further, each amino acid type is determined by the underlying DNA sequence in a gene
- The sequence of amino acids influence how the proteins fold, which dictates the function of the protein
- Protein function is a vast area of research in biotechnology, and understanding this further is critical for developing therapeutics and precision diagnostics 

<img src= "https://www.ebi.ac.uk/training/online/courses/protein-classification-intro-ebi-resources/wp-content/uploads/sites/96/2020/07/figure1.png" width=400>

<a href="https://www.ebi.ac.uk/training/online/courses/protein-classification-intro-ebi-resources/protein-classification/" target="_blank">image source</a>

### Purpose
- Proteins can be classified by their physical properties; but these classifications tend to be very general
- Amino acids each have unique physical and chemical properties, so when we have this kind of data, we can make broad generalizations of the protein function overall
- More specific classifications can be made, however, by analyzing the amino acid sequence itself
- The order of each amino acid within the longer chain is crucial for protein function, and other arrangements of these building blocks results in a totally different protein with different function

*** 
- Some examples of how these amino acid building blocks influence protein function:
    - Active sites on proteins contain amino acids involved in catalytic activity
        - Example: Lipase catalyses the formation and hydrolysis of fats --> has two amino acid residues (a histidine followed by a glycine) that are essential for its catalytic activity
    - Binding sites contain amino acids that are directly involved in binding molecules or ions
        - Example: Iron-binding site of haemoglobin
    - Post-translational modification (PTM) sites contain residues known to be chemically modified (phosphorylated, palmitoylated, acetylated, etc) after the process of protein translation
    - Repeats are typically short amino acid sequences that are repeated within a protein
***

### Dataset
- We have 2 .csv files of data: 
    - `pdb_data_no_dups.csv` provides physical properties of the protein
        - properties (numeric): residue count, pH, crystallization temperature, resolution, molecular weight, and density
            - residue count: number of amino acids in the sequence
            - pH: scale determining acidity or basicity of a solution (e.g. acidic solutions have a pH < 7.0 and basic solutions have pH > 7.0)
            - crystallization temperature: temperature at which the protein crystallizes (precipitates out of solution during crystallization process)
            - resolution: measure of the quality of the data that has been collected on the crystal containing the protein
            - molecular weight: molecular mass of a given molecule (expressed in kilo Daltons, kDa)
            - density: 
        - target label is provided here, as "classification" of the protein type (e.g. ligase, transferase, oxidoreductase, etc.)
    - `pdb_data_seq.csv` provides the amino acid sequence for each protein
    
    #### target label:
    - `classification`

    #### categorical vars: 
    - `experimentalTechnique`
    - `crystallizationMethod`
    - `pdbxDetails`
    - `publicationYear`
    - `sequence`

    #### numerical vars:
    - `residueCount`
    - `resolution`
    - `structureMolecularWeight`
    - `crystallizationTempK`
    - `densityMatthews`
    - `densityPercentSol`
    - `phValue`


### Problem Statement
- Classify the protein type/function using information provided in the dataset
- Evaluate performance of models

### Conclusions
- RNN LSTM model performed the best:
```
model_4_top10 = Sequential()
model_4_top10.add(Embedding(25, 10, input_length=285))

model_4_top10.add(LSTM(25, return_sequences=True))

model_4_top10.add(Flatten())
model_4_top10.add(Dense(285, activation='sigmoid'))
model_4_top10.add(Dense(10, activation='softmax'))


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                                  mode='max', 
                                                  restore_best_weights=True, 
                                                  patience=5)

opt = tf.keras.optimizers.Adam(learning_rate= 0.001)

model_4_top10.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
```

<img src= "https://raw.githubusercontent.com/seetarajpara/Module5_Capstone_ProteinClassificationProject/main/output/output_126_0.png" width=400>
<img src= "https://raw.githubusercontent.com/seetarajpara/Module5_Capstone_ProteinClassificationProject/main/output/output_126_1.png" width=400>
<img src= "https://raw.githubusercontent.com/seetarajpara/Module5_Capstone_ProteinClassificationProject/main/output/output_128_0.png" width=400>

```                               
                               precision    recall  f1-score   support
                    HYDROLASE       0.80      0.83      0.81      4030
HYDROLASE/HYDROLASE INHIBITOR       0.59      0.51      0.55       516
                IMMUNE SYSTEM       0.88      0.81      0.84       780
                    ISOMERASE       0.77      0.73      0.75       507
                        LYASE       0.81      0.80      0.80       880
               OXIDOREDUCTASE       0.87      0.87      0.87      2422
            SIGNALING PROTEIN       0.66      0.56      0.61       636
                TRANSCRIPTION       0.75      0.71      0.73       673
                  TRANSFERASE       0.82      0.85      0.83      3087
            TRANSPORT PROTEIN       0.76      0.72      0.74       630

                     accuracy                           0.80     14161
                    macro avg       0.77      0.74      0.75     14161
                 weighted avg       0.80      0.80      0.80     14161
```

### Sources
1. https://www.ebi.ac.uk/training/online/courses/protein-classification-intro-ebi-resources/protein-classification/
2. https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/introduction



# Repo Outline
	1) Jupyter Notebook: `SRajpara_Module5_Capstone_ProteinClassificationProject.ipynb'
	2) Image Directory: `Module5_Capstone_ProteinClassificationProject/output/` 
	3) Non-Technical Presentation: `Module5_Capstone_ProteinClassificationProject/Module 5 Capstone Presentation.pdf`
