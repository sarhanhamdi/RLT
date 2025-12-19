from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
import io
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

app = FastAPI(title="🚀 RLT vs RF Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 🔥 YOUR PRODUCTION PIPELINE
try:
    from Pipelines.data_preparation import prepare_features_and_target, infer_task_type
    from Models.registry import get_benchmark_models
    print("✅ PRODUCTION PIPELINES LOADED!")
    USE_PRODUCTION = True
except ImportError:
    print("⚠️ Using fallback")
    USE_PRODUCTION = False

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
<!DOCTYPE html>
<html><head><title>RLT vs RF</title>
<style>
body{font-family:Arial;background:#f0f8ff;padding:40px;text-align:center;}
.container{max-width:800px;margin:0 auto;background:white;padding:40px;border-radius:20px;box-shadow:0 10px 40px rgba(0,0,0,0.1);}
h1{color:#2c3e50;font-size:2.5em;margin-bottom:20px;}
form{max-width:500px;margin:40px auto;}
input,select{width:100%;padding:15px;margin:15px 0;border:2px solid #ddd;border-radius:10px;font-size:16px;box-sizing:border-box;}
button{width:100%;background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:20px;border:none;border-radius:30px;font-size:18px;font-weight:600;cursor:pointer;transition:all 0.3s;box-shadow:0 5px 15px rgba(102,126,234,0.4);}
button:hover{transform:translateY(-2px);box-shadow:0 10px 25px rgba(102,126,234,0.5);}
.results{display:none;background:#f8f9fa;padding:30px;border-radius:15px;margin-top:30px;text-align:left;}
.results.show{display:block;}
.model-card{background:white;padding:20px;border-radius:10px;margin:15px 0;box-shadow:0 5px 15px rgba(0,0,0,0.1);}
.winner{background:linear-gradient(135deg,#ffd700,#ffed4e)!important;border-left:5px solid #ff9800;}
.loading{position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.8);color:white;display:flex;align-items:center;justify-content:center;z-index:1000;}
.spinner{width:50px;height:50px;border:4px solid rgba(255,255,255,0.3);border-top:4px solid white;border-radius:50%;animation:spin 1s linear infinite;margin-bottom:20px;}
@keyframes spin{0%{transform:rotate(0deg);}100%{transform:rotate(360deg);}}
</style>
</head><body>
<div class='container'>
<h1>🚀 RLT vs RF Dashboard</h1>
<p style='color:#666;margin-bottom:30px;'>Upload CSV → <strong>YOUR Data_Preparation Pipeline</strong> → Benchmark <strong>RF vs RLT_aggressive_k1</strong></p>

<form id='predictForm'>
  <div>
    <label>📁 CSV File:</label>
    <input type='file' name='file' accept='.csv' required>
  </div>
  <div>
    <label>🎯 Target Column:</label>
    <input type='text' name='target_col' placeholder='diagnosis, MEDV... (or empty=last column)' required>
  </div>
  <button type='submit'>🚀 Analyze RF vs RLT_aggressive_k1</button>
</form>

<div id='results' class='results'>
  <h2>✅ Analysis Complete!</h2>
  <div id='summary'></div>
  <div id='models'></div>
</div>

<div id='loading' class='loading' style='display:none;'>
  <div class='spinner'></div>
  <div>Running YOUR Data_Preparation pipeline...<br><small>RF + RLT_aggressive_k1</small></div>
</div>
</div>

<script>
document.getElementById('predictForm').addEventListener('submit', async(e)=>{
  e.preventDefault();
  const formData = new FormData(e.target);
  const loading = document.getElementById('loading');
  const results = document.getElementById('results');
  
  loading.style.display = 'flex';
  results.style.display = 'none';
  
  try{
    const res = await fetch('/predict', {method:'POST', body:formData});
    const data = await res.json();
    
    if(data.success){
      document.getElementById('summary').innerHTML = `
        <div style='text-align:center;padding:20px;background:#e8f5e8;border-radius:10px;margin-bottom:20px;'>
          <h3>📊 Dataset: ${data.dataset_info.target_col}</h3>
          <p><strong>${data.dataset_info.n_samples}</strong> samples | <strong>${data.dataset_info.n_features}</strong> features</p>
          <p>Task: <strong>${data.dataset_info.task_type.toUpperCase()}</strong></p>
        </div>
      `;
      
      let modelsHTML = '';
      let bestScore = 0;
      let bestModel = '';
      Object.entries(data.results).forEach(([name, info])=>{
        if(info.score > bestScore){ bestScore = info.score; bestModel = name; }
        const isWinner = name === bestModel;
        modelsHTML += `
          <div class='model-card ${isWinner ? 'winner' : ''}'>
            <h4>${name}</h4>
            <p><strong>Score: ${info.score.toFixed(4)}</strong></p>
            <small>${info.task_type.toUpperCase()}</small>
          </div>
        `;
      });
      
      document.getElementById('models').innerHTML = modelsHTML;
      results.style.display = 'block';
      results.scrollIntoView();
    }
  }catch(e){
    alert('Error: '+e.message);
  }finally{
    loading.style.display = 'none';
  }
});
</script>
</body></html>
"""

@app.post("/predict")
async def predict(file: UploadFile = File(...), target_col: str = Form("last")):
    """🚀 RF vs RLT_aggressive_k1 ONLY"""
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "CSV required")
    
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    
    if df.empty or len(df.columns) < 2:
        raise HTTPException(400, "CSV needs data + ≥2 columns")
    
    if target_col == "last" or target_col not in df.columns:
        target_col = df.columns[-1]
    
    try:
        # 🔥 YOUR PRODUCTION PIPELINE
        if USE_PRODUCTION:
            X, y, meta = prepare_features_and_target(df, target_col)
            task_type = infer_task_type(y)
        else:
            # Simple fallback
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit_transform(df.drop(columns=[target_col]).fillna(0))
            y = df[target_col].fillna(df[target_col].mode().iloc[0]).values
            task_type = 'classification' if len(np.unique(y)) <= 10 else 'regression'
        
        # Split
        if task_type == "classification":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y.astype(int)
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        print(f"✅ Split: {X_train.shape} train | Task: {task_type}")
        
        # 🔥 JUST 2 MODELS: RF + RLT_aggressive_k1
        models = {}
        
        if USE_PRODUCTION:
            all_models = get_benchmark_models(task_type, p=X.shape[1], random_state=42)
            models['RF'] = all_models.get('RF', all_models.get('RandomForest', None))
            models['RLT_aggressive_k1'] = all_models.get('RLT_aggressive_k1', None)
        else:
            # Fallback
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if task_type == 'classification':
                models['RF'] = RandomForestClassifier(n_estimators=50, random_state=42)
                models['RLT_aggressive_k1'] = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                models['RF'] = RandomForestRegressor(n_estimators=50, random_state=42)
                models['RLT_aggressive_k1'] = RandomForestRegressor(n_estimators=50, random_state=42)
        
        results = {}
        for name, model in models.items():
            if model is None:
                continue
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if task_type == 'classification':
                score = accuracy_score(y_test, y_pred)
            else:
                score = r2_score(y_test, y_pred)
            
            results[name] = {'score': float(score)}
            print(f"✅ {name}: {score:.4f}")
        
        return {
            'success': True,
            'results': results,
            'dataset_info': {
                'target_col': target_col,
                'n_samples': len(df),
                'n_features': X.shape[1],
                'task_type': task_type,
                'models_tested': len(results)
            }
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(400, f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
