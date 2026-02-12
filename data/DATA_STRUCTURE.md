# Data Structure Guide

Place your training data in the following structure:

data/
+-- train/
    +-- acceptable/
        +-- folder_01/  # Lighting condition 1
        |   +-- glove1_light1.jpg
        |   +-- glove2_light1.jpg
        |   +-- ...
        +-- folder_02/  # Lighting condition 2
        |   +-- ...
        +-- ...
        +-- folder_16/  # Lighting condition 16

+-- test/
    +-- acceptable/
    |   +-- glove_images/
    +-- marginal/
    |   +-- glove_images/
    +-- unacceptable/
        +-- glove_images/

Each folder should contain glove images captured under a specific lighting condition.
The system will train separate models for each lighting condition.

After placing your data:
1. Run preprocessing: python main.py --preprocess
2. Verify structure: python main.py --verify
3. Start training: python main.py --train --all
