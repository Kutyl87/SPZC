# SPZC

## Przykładowe wywołania modelu
```bash
python3 teacher_api.py --dataset imagenette
python3 train.py --dataset imagenette --model resnet18 --epochs 10 --batch-size 128 --report-accuracies

python3 train.py --dataset cifar100 --model resnet18 --epochs 10 --batch-size 128

python3 train.py --dataset imagenette --model small_cnn --epochs 1 --batch-size 64 --report-accuracies
```

Ważne jest aby teacher_api.py był uruchomiony z tym samym datasetem co train.py!

Ponadto można użyć flagi -h lub --help w celu pełnego opisu działania programów.

Zależności został← opisane w pliku requirements.txt, można je zainstalwoać za pomocą komendy:
```bash
pip install -r requirements.txt
```

