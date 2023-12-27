## Data

### Bulk gene expression data & FACS true values
- GSE65133 (human blood)
- GSE107572 (human blood)
- GSE60424 (human blood)
- GSE237801 (mosue liver)
- GSE239996 (rat liver)

Note: Due to memory problems, some files are not registered in this repository. Please contact us if necessary.

### Marker genes information
``` 
├─marker
│  │  human_blood_domain.pkl
│  │  human_breast_CellMarker.pkl
│  │  human_liver_CellMarker.pkl
│  │  human_lung_CellMarker.pkl
│  │  mouse_liver_CellMarker.pkl
│  │  mouse_LM6_DEGs.pkl
│  │  rat_liver_classical.pkl
│  │
│  └─how_processed
│      │  marker_prep_human_blood.py
│      │  marker_prep_human_breast.py
│      │  marker_prep_human_liver.py
│      │  marker_prep_human_lung.py
│      │  marker_prep_mouse_liver.py
│      │  marker_prep_rat_liver.py
│      │
│      └─raw_info
```
Stores marker information that is also used in the sample code in the `./examples/` folder.

The marker genes processed here can be reproduced by running the python file in the `./data/marker/how_processed/` folder.
