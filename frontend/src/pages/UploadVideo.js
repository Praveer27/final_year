import React, { useState } from 'react';

function UploadVideo() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('video', selectedFile);

    try {
      const response = await fetch('/api/predict/video', {
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
      <h1 className="page-title">🎥 Upload Video</h1>
      <p className="page-subtitle">Upload a video of sign language gestures for analysis</p>

      <div className="upload-area" onClick={() => document.getElementById('videoInput').click()}>
        {selectedFile ? (
          <div>
            <div style={{fontSize: '4rem'}}>🎬</div>
            <p style={{fontSize: '1.2rem', marginTop: '1rem'}}>{selectedFile.name}</p>
            <p style={{color: '#666'}}>Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
          </div>
        ) : (
          <div>
            <div style={{fontSize: '4rem'}}>🎥</div>
            <p style={{fontSize: '1.2rem', marginTop: '1rem'}}>Click to select a video</p>
            <p style={{color: '#666'}}>or drag and drop here</p>
          </div>
        )}
      </div>

      <input
        id="videoInput"
        type="file"
        accept="video/*"
        onChange={handleFileSelect}
        style={{display: 'none'}}
      />

      {selectedFile && (
        <div className="flex-center gap-2 mt-3">
          <button onClick={handleUpload} className="btn btn-primary" disabled={loading}>
            {loading ? 'Processing...' : 'Analyze Video'}
          </button>
          <button onClick={() => {setSelectedFile(null); setResult(null);}} className="btn btn-secondary">
            Clear
          </button>
        </div>
      )}

      {loading && (
        <div className="mt-3">
          <div className="loading-spinner"></div>
          <p className="mt-2">Processing video... This may take a few moments.</p>
        </div>
      )}

      {result && (
        <div className="result-container">
          <h2>Analysis Result</h2>
          <div className="result-prediction">{result.prediction || 'Processing...'}</div>
          {result.confidence !== undefined && (
            <div className="result-confidence">
              Confidence: {(result.confidence * 100).toFixed(1)}%
            </div>
          )}
          {result.frames_processed && (
            <p className="mt-2">Frames Processed: {result.frames_processed}</p>
          )}
          {result.message && (
            <p className="mt-2" style={{fontSize: '0.9rem', opacity: 0.8}}>{result.message}</p>
          )}
        </div>
      )}

      <div className="mt-4" style={{background: 'white', padding: '1.5rem', borderRadius: '15px'}}>
        <h3>Video Requirements:</h3>
        <ul style={{textAlign: 'left', maxWidth: '600px', margin: '1rem auto'}}>
          <li>Supported formats: MP4, AVI, MOV, MKV</li>
          <li>Maximum file size: 100 MB</li>
          <li>Clear visibility of hand gestures</li>
          <li>Good lighting conditions</li>
          <li>Stable camera position</li>
        </ul>
        <p className="mt-2"><strong>Note:</strong> Video processing may take longer depending on video length and quality.</p>
      </div>
    </div>
  );
}

export default UploadVideo;

// Made with Bob
