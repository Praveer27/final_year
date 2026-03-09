import React, { useState } from 'react';

function UploadImage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('/api/predict/image', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      alert('Error: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container">
      <h1 className="page-title">📤 Upload Image</h1>
      <p className="page-subtitle">Upload an image of a sign language gesture for prediction</p>

      <div className="upload-area" onClick={() => document.getElementById('fileInput').click()}>
        {preview ? (
          <img src={preview} alt="Preview" style={{maxWidth: '100%', maxHeight: '400px', borderRadius: '10px'}} />
        ) : (
          <div>
            <div style={{fontSize: '4rem'}}>📁</div>
            <p style={{fontSize: '1.2rem', marginTop: '1rem'}}>Click to select an image</p>
            <p style={{color: '#666'}}>or drag and drop here</p>
          </div>
        )}
      </div>

      <input
        id="fileInput"
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        style={{display: 'none'}}
      />

      {selectedFile && (
        <div className="flex-center gap-2 mt-3">
          <button onClick={handleUpload} className="btn btn-primary" disabled={loading}>
            {loading ? 'Processing...' : 'Predict Gesture'}
          </button>
          <button onClick={() => {setSelectedFile(null); setPreview(null); setResult(null);}} className="btn btn-secondary">
            Clear
          </button>
        </div>
      )}

      {loading && <div className="loading-spinner"></div>}

      {result && (
        <div className="result-container">
          <h2>Prediction Result</h2>
          <div className="result-prediction">{result.prediction || 'Processing...'}</div>
          {result.confidence !== undefined && (
            <div className="result-confidence">
              Confidence: {(result.confidence * 100).toFixed(1)}%
            </div>
          )}
          {result.hands_detected !== undefined && (
            <p className="mt-2">Hands Detected: {result.hands_detected}</p>
          )}
          {result.message && (
            <p className="mt-2" style={{fontSize: '0.9rem', opacity: 0.8}}>{result.message}</p>
          )}
        </div>
      )}

      <div className="mt-4" style={{background: 'white', padding: '1.5rem', borderRadius: '15px'}}>
        <h3>Tips for Best Results:</h3>
        <ul style={{textAlign: 'left', maxWidth: '600px', margin: '1rem auto'}}>
          <li>Use clear, well-lit images</li>
          <li>Ensure the hand gesture is clearly visible</li>
          <li>Avoid cluttered backgrounds</li>
          <li>Supported formats: JPG, PNG, JPEG</li>
        </ul>
      </div>
    </div>
  );
}

export default UploadImage;

// Made with Bob
