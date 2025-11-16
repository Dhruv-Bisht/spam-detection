// static/script.js - Fixed version
document.addEventListener('DOMContentLoaded', function() {
    console.log('SpamGuard initialized');

    const messageInput = document.getElementById('messageInput');
    const checkBtn = document.getElementById('checkBtn');
    const clearBtn = document.getElementById('clearBtn');
    const loading = document.getElementById('loading');
    const resultBox = document.getElementById('resultBox');
    const confidenceScore = document.getElementById('confidenceScore');
    const meterFill = document.getElementById('meterFill');
    const featuresSection = document.getElementById('featuresSection');
    const insightsGrid = document.getElementById('insightsGrid');
    const charCount = document.getElementById('charCount');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const historyItems = document.getElementById('historyItems');

    // Character count functionality
    if (messageInput && charCount) {
        messageInput.addEventListener('input', function() {
            const length = this.value.length;
            charCount.textContent = length;
            checkBtn.disabled = length === 0;
        });
        charCount.textContent = messageInput.value.length;
        checkBtn.disabled = messageInput.value.length === 0;
    }

    // Reset form function
    function resetForm() {
        if (messageInput) messageInput.value = '';
        if (charCount) charCount.textContent = '0';
        if (checkBtn) checkBtn.disabled = true;
        
        if (resultBox) {
            resultBox.innerHTML = `
                <div class="result-icon">📝</div>
                <div class="result-content">
                    <div class="result-label">Enter a message to analyze</div>
                    <div class="result-desc">The AI will analyze your message for spam patterns</div>
                </div>
            `;
            resultBox.className = 'result-box';
        }
        
        if (confidenceScore) confidenceScore.textContent = '0%';
        if (meterFill) {
            meterFill.style.width = '0%';
            meterFill.className = 'meter-fill';
        }
        if (featuresSection) featuresSection.style.display = 'none';
        if (loading) loading.style.display = 'none';
    }

    // Analyze button
    if (checkBtn && messageInput) {
        checkBtn.addEventListener('click', async function(e) {
            e.preventDefault();
            
            const text = messageInput.value.trim();
            if (!text) {
                alert('Please enter a message to analyze.');
                return;
            }

            console.log('🔍 Analyzing text:', text);

            // Show loading
            if (loading) loading.style.display = 'flex';
            checkBtn.disabled = true;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });

                console.log('📡 Response status:', response.status);

                if (!response.ok) {
                    throw new Error(`Server returned ${response.status}`);
                }

                const data = await response.json();
                console.log('📊 Prediction data:', data);

                // Hide loading
                if (loading) loading.style.display = 'none';

                const isSpam = data.prediction === 'spam';
                const confPercent = data.confidence;
                const confValue = parseFloat(confPercent.replace('%', '')) / 100;

                // Update result box
                if (resultBox) {
                    const icon = isSpam ? '🛑' : '✅';
                    const label = isSpam ? 'SPAM DETECTED' : 'LEGITIMATE MESSAGE';
                    const desc = isSpam ? 
                        'This message contains suspicious patterns typical of spam.' : 
                        'No spam patterns detected. This message appears legitimate.';
                    
                    resultBox.innerHTML = `
                        <div class="result-icon">${icon}</div>
                        <div class="result-content">
                            <div class="result-label">${label}</div>
                            <div class="result-desc">${desc}</div>
                        </div>
                    `;
                    resultBox.className = isSpam ? 'result-box spam' : 'result-box ham';
                }

                // Update confidence score
                if (confidenceScore) confidenceScore.textContent = confPercent;

                // Update meter
                if (meterFill) {
                    meterFill.style.width = `${confValue * 100}%`;
                    meterFill.className = isSpam ? 'meter-fill spam' : 'meter-fill ham';
                }

                // Update insights
                if (insightsGrid && featuresSection) {
                    const insights = isSpam ? [
                        { icon: '⚠️', text: 'Urgent or promotional language detected' },
                        { icon: '💰', text: 'Financial or prize-related content' },
                        { icon: '🔗', text: 'Suspicious calls-to-action found' },
                        { icon: '🚩', text: 'Multiple spam indicators identified' }
                    ] : [
                        { icon: '👍', text: 'Natural communication style' },
                        { icon: '📧', text: 'Professional or personal tone' },
                        { icon: '🔒', text: 'No suspicious patterns found' },
                        { icon: '✅', text: 'Clean message content' }
                    ];
                    
                    insightsGrid.innerHTML = insights.map(insight => `
                        <div class="insight-item">
                            <div class="insight-icon">${insight.icon}</div>
                            <div class="insight-text">${insight.text}</div>
                        </div>
                    `).join('');
                    featuresSection.style.display = 'block';
                }

                // Add to history
                addToHistory(
                    text.substring(0, 100) + (text.length > 100 ? '...' : ''), 
                    data.prediction, 
                    confPercent, 
                    new Date().toLocaleString()
                );

            } catch (error) {
                console.error('❌ Analysis error:', error);
                
                // Hide loading
                if (loading) loading.style.display = 'none';
                checkBtn.disabled = false;

                // Show error message
                if (resultBox) {
                    resultBox.innerHTML = `
                        <div class="result-icon">❌</div>
                        <div class="result-content">
                            <div class="result-label">ANALYSIS ERROR</div>
                            <div class="result-desc">Unable to analyze message. Please try again.</div>
                        </div>
                    `;
                    resultBox.className = 'result-box error';
                }
                
                if (confidenceScore) confidenceScore.textContent = '0%';
                if (meterFill) {
                    meterFill.style.width = '0%';
                    meterFill.className = 'meter-fill';
                }
                if (featuresSection) featuresSection.style.display = 'none';

                // Add error to history
                addToHistory(
                    text.substring(0, 100) + (text.length > 100 ? '...' : ''),
                    'error',
                    '0%',
                    new Date().toLocaleString()
                );
            }
        });
    }

    // Clear button
    if (clearBtn) {
        clearBtn.addEventListener('click', function() {
            resetForm();
        });
    }

    // History functionality
    function addToHistory(message, prediction, confidence, timestamp) {
        let history = JSON.parse(localStorage.getItem('spamHistory') || '[]');
        history.unshift({ message, prediction, confidence, timestamp });
        if (history.length > 10) history = history.slice(0, 10);
        localStorage.setItem('spamHistory', JSON.stringify(history));
        loadHistory();
    }

    // In the loadHistory() function, replace the history item generation:

function loadHistory() {
    const history = JSON.parse(localStorage.getItem('spamHistory') || '[]');
    historyItems.innerHTML = history.map(item => {
        let className = 'history-item';
        let badgeClass = '';
        let badgeText = item.prediction.toUpperCase();
        
        if (item.prediction === 'spam') {
            className += ' spam';
            badgeClass = 'spam';
        } else if (item.prediction === 'not spam') {
            className += ' ham';
            badgeClass = 'ham';
            badgeText = 'LEGITIMATE';
        } else {
            className += ' error';
            badgeClass = 'error';
            badgeText = 'ERROR';
        }
        
        return `
            <div class="${className}">
                <div class="history-content">
                    <div class="history-header">
                        <div class="history-main-info">
                            <span class="prediction-badge ${badgeClass}">${badgeText}</span>
                            <span class="confidence-display">${item.confidence}</span>
                        </div>
                    </div>
                    <div class="history-message">${item.message}</div>
                </div>
            </div>
        `;
    }).join('') || '<p>No history yet. Analyze some messages!</p>';
}
    function clearHistory() {
        localStorage.removeItem('spamHistory');
        loadHistory();
    }

    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', function(e) {
            e.preventDefault();
            if (confirm('Clear all detection history?')) {
                clearHistory();
            }
        });
    }

    // Load initial history
    loadHistory();
    
    // Test the connection on load
    console.log('✅ SpamGuard frontend loaded successfully');
});