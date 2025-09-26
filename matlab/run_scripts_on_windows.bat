@echo off
REM === Sequential execution of each MATLAB training script ===

matlab -batch "setup_env('C:\Users\marco_u3rv1hf\AppData\Local\Programs\Python\Python39\python', 'C:\Users\marco_u3rv1hf\Desktop\Unifi\Borse_di_ricerca\00_AImetrics_Verdecchia\Progetto\DeepGreen'); resnet18.train_cifar100; exit"
timeout /t 30 /nobreak

matlab -batch "setup_env('C:\Users\marco_u3rv1hf\AppData\Local\Programs\Python\Python39\python', 'C:\Users\marco_u3rv1hf\Desktop\Unifi\Borse_di_ricerca\00_AImetrics_Verdecchia\Progetto\DeepGreen'); resnet18.train_fashion; exit"
timeout /t 30 /nobreak

matlab -batch "setup_env('C:\Users\marco_u3rv1hf\AppData\Local\Programs\Python\Python39\python', 'C:\Users\marco_u3rv1hf\Desktop\Unifi\Borse_di_ricerca\00_AImetrics_Verdecchia\Progetto\DeepGreen'); resnet18.train_tiny; exit"
timeout /t 30 /nobreak

matlab -batch "setup_env('C:\Users\marco_u3rv1hf\AppData\Local\Programs\Python\Python39\python', 'C:\Users\marco_u3rv1hf\Desktop\Unifi\Borse_di_ricerca\00_AImetrics_Verdecchia\Progetto\DeepGreen'); vgg16.train_cifar100; exit"
timeout /t 30 /nobreak

matlab -batch "setup_env('C:\Users\marco_u3rv1hf\AppData\Local\Programs\Python\Python39\python', 'C:\Users\marco_u3rv1hf\Desktop\Unifi\Borse_di_ricerca\00_AImetrics_Verdecchia\Progetto\DeepGreen'); vgg16.train_fashion; exit"
timeout /t 30 /nobreak

matlab -batch "setup_env('C:\Users\marco_u3rv1hf\AppData\Local\Programs\Python\Python39\python', 'C:\Users\marco_u3rv1hf\Desktop\Unifi\Borse_di_ricerca\00_AImetrics_Verdecchia\Progetto\DeepGreen'); vgg16.train_tiny; exit"

echo All scripts have been executed.
pause
