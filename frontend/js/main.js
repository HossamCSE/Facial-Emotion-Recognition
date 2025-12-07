/* 
============================================================
 Facial Emotion Recognition - Frontend Main Logic
 -----------------------------------------------------------
 This file handles:
 1. Camera capture (Start / Capture / Reset)
 2. Upload image preview + drag & drop
 3. Sending Base64 image to Flask backend
 4. Receiving predictions + displaying results
 5. Updating statistics and charts (Doughnut Chart)
 
============================================================
*/


// --- API Base URL ---
const API_BASE = '/api';  // Relative path as frontend is served by Flask

// --- Global Variables ---
let stream = null;  // Camera stream object

// --- DOM Elements ---
const video = document.getElementById('videoElement');          // Video preview element
const canvas = document.getElementById('canvasElement');       // Canvas to capture frames
const resultImgLive = document.getElementById('resultImageLive'); // Image element to show processed live frame
const fileInput = document.getElementById('fileInput');        // File input for uploads
const loadingScreen = document.getElementById('loading-screen'); // Loading overlay

// ==================================================
// Initialization

document.addEventListener('DOMContentLoaded', () => {
    refreshStats();       // Load stats on page load
    setupEventListeners(); // Setup all buttons, inputs, and drag/drop
    switchTab('live');    // Default to live camera tab
});

// ==================================================
// Event Listeners Setup

function setupEventListeners() {
    // --- Camera Controls ---
    document.getElementById('startBtn').addEventListener('click', startCamera);
    document.getElementById('captureBtn').addEventListener('click', captureAndAnalyze);
    document.getElementById('resetLiveBtn').addEventListener('click', resetCameraView);

    // --- Upload Controls ---
    const dropZone = document.getElementById('dropZone');
    
    // Click to browse
    dropZone.addEventListener('click', () => fileInput.click());
    
    // File selection
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag & Drop styling
    dropZone.addEventListener('dragover', (e) => { 
        e.preventDefault(); 
        dropZone.style.borderColor = '#fff'; 
    });
    dropZone.addEventListener('dragleave', (e) => { 
        e.preventDefault(); 
        dropZone.style.borderColor = 'var(--secondary-color)'; 
    });
    
    // Drop file
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        if(e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    // Analyze uploaded image
    document.getElementById('analyzeUploadBtn').addEventListener('click', () => {
        const img = document.getElementById('uploadedImageDisplay');
        sendToAPI(img.src, 'upload'); // Pass 'upload' as source
    });

    // Reset upload view
    document.getElementById('newUploadBtn').addEventListener('click', resetUploadView);
}

// ==================================================
// Tab Switching Logic

function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(el => {
        el.style.display = 'none';
        el.classList.remove('active');
    });
    
    // Remove active class from nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));
    
    if (tabName === 'live') {
        const liveTab = document.getElementById('live-tab');
        liveTab.style.display = 'block';
        liveTab.classList.add('active');
        document.querySelector('button[onclick="switchTab(\'live\')"]').classList.add('active');
    } else {
        stopCamera(); // Stop camera to save resources
        const uploadTab = document.getElementById('upload-tab');
        uploadTab.style.display = 'block';
        uploadTab.classList.add('active');
        document.querySelector('button[onclick="switchTab(\'upload\')"]').classList.add('active');
    }
}

// ==================================================
// Camera Logic

async function startCamera() {
    try {
        // Request camera access
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.classList.remove('hidden');
        resultImgLive.classList.add('hidden');
        
        // Button visibility
        document.getElementById('startBtn').classList.add('hidden');
        document.getElementById('captureBtn').disabled = false;
        document.getElementById('resetLiveBtn').classList.add('hidden');
    } catch (err) {
        alert("Error accessing camera. Please allow permissions.");
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        video.srcObject = null;
    }
}

function captureAndAnalyze() {
    // Capture frame
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    const imageData = canvas.toDataURL('image/jpeg'); // Convert to Base64

    stopCamera(); // Stop preview
    video.classList.add('hidden');

    sendToAPI(imageData, 'live');
}

function resetCameraView() {
    resultImgLive.classList.add('hidden');
    video.classList.remove('hidden');
    document.getElementById('resetLiveBtn').classList.add('hidden');
    startCamera();
}

// ==================================================
// Upload Logic

function handleFileSelect(e) {
    if (e.target.files.length) handleFile(e.target.files[0]);
}

function handleFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('uploadedImageDisplay').src = e.target.result;
        document.getElementById('dropZone').classList.add('hidden');
        document.getElementById('uploadPreview').classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

function resetUploadView() {
    document.getElementById('uploadPreview').classList.add('hidden');
    document.getElementById('dropZone').classList.remove('hidden');
    fileInput.value = '';
}

// ==================================================
// API Integration

async function sendToAPI(base64Image, source) {
    showLoading(true);
    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Image })
        });
        const data = await response.json();
        
        if (data.success) {
            displayResults(data, source);
            refreshStats();
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        console.error(error);
        alert('Server connection failed.');
    } finally {
        showLoading(false);
    }
}

function displayResults(data, source) {
    // Show processed image
    if (source === 'live') {
        resultImgLive.src = data.image;
        resultImgLive.classList.remove('hidden');
        document.getElementById('resetLiveBtn').classList.remove('hidden');
        document.getElementById('captureBtn').disabled = true;
    } else {
        document.getElementById('uploadedImageDisplay').src = data.image;
    }

    // Update sidebar results
    const container = document.getElementById('results-container');
    container.innerHTML = '';
    
    if (!data.faces || data.faces.length === 0) {
        container.innerHTML = '<div class="empty-state"><p>No faces detected.</p></div>';
        return;
    }

    data.faces.forEach(face => {
        const el = document.createElement('div');
        el.className = 'result-item';
        el.innerHTML = `
            <h4>${face.emotion}</h4>
            <small>Confidence: ${face.confidence}%</small>
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${face.confidence}%"></div>
            </div>
        `;
        container.appendChild(el);
    });
}

// ==================================================
// Stats & Charts

async function refreshStats() {
    try {
        const res = await fetch(`${API_BASE}/stats`);
        const data = await res.json();
        
        if (data.success) {
            document.getElementById('total-scans').innerText = data.total;
            
            let confPercent = data.avg_confidence ? (data.avg_confidence * 100).toFixed(1) : 0;
            document.getElementById('avg-conf').innerText = confPercent + '%';
            
            renderChart(data.emotions);
        }
    } catch (e) { console.error("Stats error", e); }
}

let myChart = null;
function renderChart(emotions) {
    const ctx = document.getElementById('emotionChart').getContext('2d');
    if (myChart) myChart.destroy();

    if (!emotions || Object.keys(emotions).length === 0) return; // Nothing to plot

    myChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(emotions),
            datasets: [{
                data: Object.values(emotions),
                backgroundColor: ['#ff7675', '#74b9ff', '#55efc4', '#ffeaa7', '#a29bfe', '#fd79a8'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'right', labels: { color: 'white', boxWidth: 12 } }
            }
        }
    });
}

// ==================================================
// Loading Screen

function showLoading(show) {
    loadingScreen.classList.toggle('hidden', !show);
}

// ==================================================
// Scroll to Stats Section

function scrollToStats() {
    document.getElementById('stats-section').scrollIntoView({ behavior: 'smooth' });
}

// ==================================================
// Clear Prediction History

async function clearData() {
    if (!confirm("Are you sure you want to delete all prediction history?")) return;

    try {
        const res = await fetch(`${API_BASE}/clear`, { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            alert("History cleared!");
            refreshStats(); // Reset UI
        }
    } catch (e) {
        alert("Failed to clear data.");
    }
}
