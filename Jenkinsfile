pipeline {
  agent any

  stages {
    stage('Checkout') {
      steps { checkout scm }
    }

    stage('Install Dependencies') {
      steps {
        bat '''
          python -m pip install --upgrade pip
          python -m pip install -r service\\requirements.txt
        '''
      }
    }

    stage('Unit Tests') {
      steps {
        bat '''
          python -m pytest -q
        '''
      }
    }

    stage('Train Model Artifact') {
      steps {
        bat '''
          python service\\train_local.py
          dir artifacts
          type artifacts\\metrics.txt
        '''
      }
    }

    stage('Build Docker Image') {
      steps {
        bat '''
          docker build -t iris-api:jenkins -f service\\Dockerfile .
        '''
      }
    }

    stage('Deploy Blue-Green') {
      steps {
        bat '''
          docker compose down
          docker compose up -d
          docker compose ps
        '''
      }
    }

    stage('Smoke Test') {
      steps {
        bat '''
          curl http://127.0.0.1:8081/health
          curl -X POST http://127.0.0.1:8081/predict -H "Content-Type: application/json" -d "{\\"features\\":[5.1,3.5,1.4,0.2]}"
        '''
      }
    }
  }
}
