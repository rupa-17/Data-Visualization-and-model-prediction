from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
import os
import subprocess
from django.conf import settings
from django.http import JsonResponse
from MyApp.Code.PythonCodes.preprocess import preprocess
from MyApp.Code.PythonCodes.dataanalysis import analysis
from MyApp.Code.PythonCodes.supervised import supervised_learning
from MyApp.Code.PythonCodes.unsupervised import unsupervised_learning
from MyApp.Code.PythonCodes.regression import regression_models
from MyApp.Code.PythonCodes.classification import classification_models
from MyApp.Code.PythonCodes.clustering import clustering_models


import json
result = None

# Create your views here.

def home(request) :
    return render(request, 'home.html')

def login_user(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, "Login successful! Welcome, " + username)
            return redirect('index')
        else:
            messages.error(request, "No account found or incorrect password. Try again!")
            return redirect('login')

    return render(request, 'login2.html')

def register(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists! Try another.")
            return redirect('register')

        user = User.objects.create_user(username=username, password=password)
        user.save()
        messages.success(request, "User registered successfully! Please log in.")
        return redirect('login')

    return render(request, 'register2.html')

def index(request) :

    return render(request, 'index.html')

def upload(request):

    global result
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        action_type = request.POST.get('action_type', '')  # Get action type
        print("Action Type Received:", action_type)

        # File storage setup
        upload_dir = os.path.join(settings.MEDIA_ROOT, "datasets")  
        os.makedirs(upload_dir, exist_ok=True)

        fs = FileSystemStorage(location=upload_dir, base_url=f"{settings.MEDIA_URL}datasets/")
        filesaved = fs.save(uploaded_file.name, uploaded_file)
        full_path = os.path.join(upload_dir, filesaved)
        file_url = fs.base_url + filesaved

        df = preprocess(full_path)

        if df is not None:
            df_data = df.to_dict(orient="records")
            df_keys = list(df.columns)
        else:
            df_data, df_keys = [], []

        df_types = {}
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df_types[col] = 'numeric'
            else:
                df_types[col] = 'categorical'


        context = {
            "file_url": file_url,
            "df_data": json.dumps(df_data),
            "df_keys": df_keys,
            "df_types" : json.dumps(df_types)
        }

        # Redirect based on button clicked
        if action_type == "visualization":
            return render(request, 'data_visualization.html', context)  # Render Data Visualization page
        elif action_type == "prediction":
            result = analysis(context)

            # supervised_acc = supervised_learning(result)
            # unsupervised_acc = unsupervised_learning(result)

            # print("Supervised ")
            # print(supervised_acc)

            # print("UnSupervised ")
            # print(unsupervised_acc)
            return render(request, 'model_prediction.html', context)  # Render Model Prediction page
        else:
            print("Invalid action_type or not received")

    return render(request, 'index.html')

def second(request) :
    
    return render(request, 'second.html')

def data_visualization(request) :

    return render(request, 'data_visualization.html')

def model_prediction(request) :

    return render(request, 'model_prediction.html')

def load_model_section(request):
    global result
    model_type = request.GET.get('type')
    results = None

    # You can call different functions based on type
    if model_type == "regression":
        title = "Regression Results"
        message = "Regression model summary goes here."
        results = regression_models(result)
         
    elif model_type == "classification":
        title = "Classification Report"
        message = "Classification metrics shown here."
        results = classification_models(result)

    elif model_type == "clustering":
        title = "Clustering Output"
        message = "Clusters and plots appear here."
        results = clustering_models(result)

    elif model_type == "neural":
        title = "Neural Networks"
        message = "Neural network layers & results shown here."
    else:
        title = "Unknown"
        message = "Invalid model type."

    return JsonResponse({
        "title": title,
        "message": message,
        "results" : results
    })
