# sst2
```
Main params:
- GPUs: 8
- max_length 128
- batch_size 32
- epochs 5
```

- **`bert-base-cased`**
- 100% train: 0.9105504587155964
- 33% (easy): 0.8772935779816514
- 33% (hard): 0.908256880733945
- 33% (ambiguous): 0.9025229357798165

---

- **`distilbert-base-cased`**
- 100% train: 0.9094036697247706
- 33% (easy): 0.856651376146789
- 33% (hard): 0.8944954128440367
- 33% (ambiguous): 0.8956422018348624

---

- **`bert-tiny`**
- 100% train: 0.788990825688
- 33% (easy): 0.694954128440367
- 33% (hard): 0.37155963302752293
- 33% (ambiguous): 0.5022935779816514

use the data selection by `distilbert-base-cased`
- 33% (easy): 0.5389908256880734
- 33% (hard): 0.5871559633027523
- 33% (ambiguous): 0.6020642201834863

