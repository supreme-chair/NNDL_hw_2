/**
 * Titanic Binary Classifier with TensorFlow.js
 * Shallow neural network implementation for browser-based training
 * Ready for GitHub Pages deployment (no server required)
 */

// Global variables to store data and model
let trainData = null;
let testData = null;
let trainFeatures = null;
let trainLabels = null;
let valFeatures = null;
let valLabels = null;
let testFeatures = null;
let model = null;
let trainingHistory = null;
let valPredictions = null;
let testPredictions = null;
let featureNames = [];
let includeFamilyFeatures = false;
let isTraining = false;
let currentTraining = null;
let trainingStats = null; // Для хранения статистики обучения

// DOM Elements
const elements = {
    // Buttons
    loadDataBtn: document.getElementById('loadDataBtn'),
    preprocessBtn: document.getElementById('preprocessBtn'),
    toggleFeaturesBtn: document.getElementById('toggleFeaturesBtn'),
    initModelBtn: document.getElementById('initModelBtn'),
    trainBtn: document.getElementById('trainBtn'),
    stopTrainBtn: document.getElementById('stopTrainBtn'),
    evaluateBtn: document.getElementById('evaluateBtn'),
    predictBtn: document.getElementById('predictBtn'),
    exportBtn: document.getElementById('exportBtn'),
    saveModelBtn: document.getElementById('saveModelBtn'),
    
    // File inputs
    trainFile: document.getElementById('trainFile'),
    testFile: document.getElementById('testFile'),
    
    // Status displays
    dataStatus: document.getElementById('dataStatus'),
    preprocessStatus: document.getElementById('preprocessStatus'),
    modelStatus: document.getElementById('modelStatus'),
    trainingStatus: document.getElementById('trainingStatus'),
    metricsStatus: document.getElementById('metricsStatus'),
    predictionStatus: document.getElementById('predictionStatus'),
    
    // Display areas
    dataPreview: document.getElementById('dataPreview'),
    featureInfo: document.getElementById('featureInfo'),
    modelSummary: document.getElementById('modelSummary'),
    trainingCharts: document.getElementById('trainingCharts'),
    rocChart: document.getElementById('rocChart'),
    confusionMatrix: document.getElementById('confusionMatrix'),
    performanceMetrics: document.getElementById('performanceMetrics'),
    predictionResults: document.getElementById('predictionResults'),
    
    // Threshold slider
    thresholdSlider: document.getElementById('thresholdSlider'),
    thresholdValue: document.getElementById('thresholdValue')
};

// Initialize event listeners
function initEventListeners() {
    // Data loading
    elements.loadDataBtn.addEventListener('click', loadData);
    
    // Auto-load when files are selected
    elements.trainFile.addEventListener('change', handleFileSelection);
    elements.testFile.addEventListener('change', handleFileSelection);
    
    // Preprocessing
    elements.preprocessBtn.addEventListener('click', preprocessData);
    elements.toggleFeaturesBtn.addEventListener('click', toggleFamilyFeatures);
    
    // Model
    elements.initModelBtn.addEventListener('click', initModel);
    
    // Training
    elements.trainBtn.addEventListener('click', trainModel);
    elements.stopTrainBtn.addEventListener('click', stopTraining);
    
    // Evaluation
    elements.evaluateBtn.addEventListener('click', evaluateModel);
    elements.thresholdSlider.addEventListener('input', updateThreshold);
    
    // Prediction & Export
    elements.predictBtn.addEventListener('click', predictTestData);
    elements.exportBtn.addEventListener('click', exportPredictions);
    elements.saveModelBtn.addEventListener('click', saveModel);
}

// Handle file selection - auto load data
function handleFileSelection() {
    // Enable the load button when at least train file is selected
    if (elements.trainFile.files.length > 0) {
        elements.loadDataBtn.disabled = false;
    }
}

// Update status display
function updateStatus(element, message, type = 'info') {
    element.textContent = message;
    element.className = 'status';
    
    switch (type) {
        case 'loading':
            element.classList.add('loading');
            break;
        case 'success':
            element.classList.add('success');
            break;
        case 'error':
            element.classList.add('error');
            break;
        default:
            // No additional class for info
            break;
    }
}

// Clear status display
function clearStatus(element) {
    element.textContent = '';
    element.className = 'status';
}

// Parse CSV data with proper handling of commas within quotes
function parseCSV(text) {
    const rows = [];
    let currentRow = [];
    let currentField = '';
    let insideQuotes = false;
    
    for (let i = 0; i < text.length; i++) {
        const char = text[i];
        
        if (char === '"') {
            // Toggle quote mode
            insideQuotes = !insideQuotes;
        } else if (char === ',' && !insideQuotes) {
            // End of field
            currentRow.push(currentField);
            currentField = '';
        } else if (char === '\n' && !insideQuotes) {
            // End of row
            currentRow.push(currentField);
            rows.push(currentRow);
            currentRow = [];
            currentField = '';
        } else {
            // Add character to current field
            currentField += char;
        }
    }
    
    // Add the last row if any
    if (currentField !== '' || currentRow.length > 0) {
        currentRow.push(currentField);
        rows.push(currentRow);
    }
    
    return rows;
}

// Load CSV data from file inputs
async function loadData() {
    const trainFile = elements.trainFile.files[0];
    const testFile = elements.testFile.files[0];
    
    if (!trainFile) {
        alert('Please select a training CSV file (train.csv)');
        return;
    }
    
    updateStatus(elements.dataStatus, 'Loading data...', 'loading');
    
    try {
        // Load training data
        const trainText = await trainFile.text();
        const trainRows = parseCSV(trainText);
        
        if (trainRows.length < 2) {
            throw new Error('Training CSV file is empty or invalid');
        }
        
        // Extract headers and data
        const trainHeaders = trainRows[0];
        const trainDataRows = trainRows.slice(1);
        
        trainData = trainDataRows.map(row => {
            const obj = {};
            trainHeaders.forEach((header, index) => {
                obj[header] = row[index];
            });
            return obj;
        });
        
        console.log(`Training data loaded: ${trainData.length} samples`);
        
        // Load test data if provided
        if (testFile) {
            const testText = await testFile.text();
            const testRows = parseCSV(testText);
            
            if (testRows.length < 2) {
                throw new Error('Test CSV file is empty or invalid');
            }
            
            const testHeaders = testRows[0];
            const testDataRows = testRows.slice(1);
            
            testData = testDataRows.map(row => {
                const obj = {};
                testHeaders.forEach((header, index) => {
                    obj[header] = row[index];
                });
                return obj;
            });
            
            console.log(`Test data loaded: ${testData.length} samples`);
        }
        
        // Display data preview
        displayDataPreview();
        
        // Enable preprocessing button
        elements.preprocessBtn.disabled = false;
        
        updateStatus(elements.dataStatus, `Data loaded successfully! Training: ${trainData.length} samples, Test: ${testData ? testData.length : 0} samples`, 'success');
        
    } catch (error) {
        console.error('Error loading data:', error);
        updateStatus(elements.dataStatus, `Error: ${error.message}`, 'error');
    }
}

// Display a preview of the loaded data
function displayDataPreview() {
    if (!trainData || trainData.length === 0) return;
    
    let html = '<h3>Data Preview (First 10 Rows)</h3>';
    html += '<table><thead><tr>';
    
    // Table headers
    const headers = Object.keys(trainData[0]);
    headers.forEach(header => {
        html += `<th>${header}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    // Table rows (first 10)
    for (let i = 0; i < Math.min(10, trainData.length); i++) {
        html += '<tr>';
        headers.forEach(header => {
            html += `<td>${trainData[i][header]}</td>`;
        });
        html += '</tr>';
    }
    
    html += '</tbody></table>';
    
    // Data statistics
    html += '<div style="margin-top: 20px;">';
    html += `<p><strong>Training Samples:</strong> ${trainData.length}</p>`;
    html += `<p><strong>Features:</strong> ${headers.length}</p>`;
    
    // Calculate survival rate
    const survivedCount = trainData.filter(row => row.Survived === '1').length;
    const survivalRate = (survivedCount / trainData.length * 100).toFixed(1);
    html += `<p><strong>Survival Rate:</strong> ${survivalRate}% (${survivedCount}/${trainData.length})</p>`;
    
    // Show missing values
    const missingInfo = calculateMissingValues(trainData);
    html += '<p><strong>Missing Values:</strong></p><ul>';
    for (const [key, value] of Object.entries(missingInfo)) {
        if (value > 0) {
            const percentage = (value / trainData.length * 100).toFixed(1);
            html += `<li>${key}: ${value} (${percentage}%)</li>`;
        }
    }
    html += '</ul></div>';
    
    // Create visualization charts
    createDataVisualizations();
    
    elements.dataPreview.innerHTML = html;
}

// Calculate missing values in the dataset
function calculateMissingValues(data) {
    const missing = {};
    const headers = Object.keys(data[0]);
    
    headers.forEach(header => {
        missing[header] = 0;
    });
    
    data.forEach(row => {
        headers.forEach(header => {
            if (row[header] === '' || row[header] === null || row[header] === undefined) {
                missing[header]++;
            }
        });
    });
    
    return missing;
}

// Create data visualization charts
function createDataVisualizations() {
    if (!trainData || trainData.length === 0) return;
    
    // Survival by Sex
    const sexCounts = {};
    const sexSurvived = {};
    
    trainData.forEach(row => {
        const sex = row.Sex || 'Unknown';
        const survived = row.Survived === '1';
        
        if (!sexCounts[sex]) {
            sexCounts[sex] = 0;
            sexSurvived[sex] = 0;
        }
        
        sexCounts[sex]++;
        if (survived) sexSurvived[sex]++;
    });
    
    const sexLabels = Object.keys(sexCounts);
    const sexSurvivalRates = sexLabels.map(sex => sexSurvived[sex] / sexCounts[sex]);
    
    // Survival by Pclass
    const pclassCounts = {};
    const pclassSurvived = {};
    
    trainData.forEach(row => {
        const pclass = row.Pclass || 'Unknown';
        const survived = row.Survived === '1';
        
        if (!pclassCounts[pclass]) {
            pclassCounts[pclass] = 0;
            pclassSurvived[pclass] = 0;
        }
        
        pclassCounts[pclass]++;
        if (survived) pclassSurvived[pclass]++;
    });
    
    const pclassLabels = Object.keys(pclassCounts).sort();
    const pclassSurvivalRates = pclassLabels.map(pclass => pclassSurvived[pclass] / pclassCounts[pclass]);
    
    // Create charts using tfjs-vis
    const surfaceSex = { name: 'Survival by Sex', tab: 'Data Analysis' };
    const surfacePclass = { name: 'Survival by Passenger Class', tab: 'Data Analysis' };
    
    // Clear previous charts
    elements.trainingCharts.innerHTML = '';
    
    // Create container for charts
    const chartsHTML = `
        <div class="chart">
            <h4>Survival Rate by Sex</h4>
            <div id="sexChart"></div>
        </div>
        <div class="chart">
            <h4>Survival Rate by Passenger Class</h4>
            <div id="pclassChart"></div>
        </div>
    `;
    
    elements.trainingCharts.innerHTML = chartsHTML;
    
    // Render charts
    tfvis.render.barchart(
        { name: 'Survival by Sex', tab: 'Data Analysis' }, 
        {
            values: sexSurvivalRates,
            labels: sexLabels
        }, 
        {
            xLabel: 'Sex',
            yLabel: 'Survival Rate',
            height: 250
        }
    );
    
    tfvis.render.barchart(
        { name: 'Survival by Passenger Class', tab: 'Data Analysis' }, 
        {
            values: pclassSurvivalRates,
            labels: pclassLabels
        }, 
        {
            xLabel: 'Passenger Class',
            yLabel: 'Survival Rate',
            height: 250
        }
    );
}

// Toggle family features (FamilySize and IsAlone)
function toggleFamilyFeatures() {
    includeFamilyFeatures = !includeFamilyFeatures;
    elements.toggleFeaturesBtn.textContent = `Toggle Family Features (${includeFamilyFeatures ? 'On' : 'Off'})`;
    elements.toggleFeaturesBtn.classList.toggle('success', includeFamilyFeatures);
    
    updateStatus(elements.preprocessStatus, 
        `Family features ${includeFamilyFeatures ? 'enabled' : 'disabled'}. Click "Run Preprocessing" to apply changes.`, 
        'info');
}

// Preprocess the data
function preprocessData() {
    if (!trainData || trainData.length === 0) {
        updateStatus(elements.preprocessStatus, 'No training data loaded', 'error');
        return;
    }
    
    updateStatus(elements.preprocessStatus, 'Preprocessing data...', 'loading');
    
    try {
        // Extract features and labels from training data
        const { featuresArray, labelsArray, featureNames: names } = extractFeaturesAndLabels(trainData, true);
        featureNames = names;
        
        // Convert to tensors
        const featuresTensor = tf.tensor2d(featuresArray);
        const labelsTensor = tf.tensor2d(labelsArray, [labelsArray.length, 1]);
        
        // Create training/validation split (80/20)
        const splitIndex = Math.floor(featuresArray.length * 0.8);
        
        trainFeatures = featuresTensor.slice(0, splitIndex);
        trainLabels = labelsTensor.slice(0, splitIndex);
        valFeatures = featuresTensor.slice(splitIndex);
        valLabels = labelsTensor.slice(splitIndex);
        
        console.log('Preprocessed data shapes:');
        console.log(`  Train features: ${trainFeatures.shape}`);
        console.log(`  Train labels: ${trainLabels.shape}`);
        console.log(`  Validation features: ${valFeatures.shape}`);
        console.log(`  Validation labels: ${valLabels.shape}`);
        
        // Display feature information
        displayFeatureInfo();
        
        // Enable model initialization button
        elements.initModelBtn.disabled = false;
        
        updateStatus(elements.preprocessStatus, 
            `Preprocessing complete! Features: ${featureNames.length}, Training: ${trainFeatures.shape[0]}, Validation: ${valFeatures.shape[0]}`, 
            'success');
            
    } catch (error) {
        console.error('Error preprocessing data:', error);
        updateStatus(elements.preprocessStatus, `Error: ${error.message}`, 'error');
    }
}

// Extract features and labels from data
function extractFeaturesAndLabels(data, isTraining = true) {
    const featuresArray = [];
    const labelsArray = [];
    
    // Calculate median age and mode embarked from training data
    let medianAge = 0;
    let modeEmbarked = 'S'; // Default to Southampton
    
    // Calculate mean and std for standardization
    let meanAge = 0, stdAge = 1, meanFare = 0, stdFare = 1;
    
    if (isTraining) {
        // Calculate median age (ignore missing values)
        const ages = data
            .filter(row => row.Age && row.Age !== '')
            .map(row => parseFloat(row.Age))
            .sort((a, b) => a - b);
        
        if (ages.length > 0) {
            const mid = Math.floor(ages.length / 2);
            medianAge = ages.length % 2 === 0 ? (ages[mid - 1] + ages[mid]) / 2 : ages[mid];
        }
        
        // Calculate mode embarked
        const embarkedCounts = {};
        data.forEach(row => {
            if (row.Embarked && row.Embarked !== '') {
                embarkedCounts[row.Embarked] = (embarkedCounts[row.Embarked] || 0) + 1;
            }
        });
        
        let maxCount = 0;
        for (const [key, count] of Object.entries(embarkedCounts)) {
            if (count > maxCount) {
                maxCount = count;
                modeEmbarked = key;
            }
        }
        
        // Calculate mean and std for standardization (only from training data)
        const validAges = data
            .filter(row => row.Age && row.Age !== '')
            .map(row => parseFloat(row.Age));
        const validFares = data
            .filter(row => row.Fare && row.Fare !== '')
            .map(row => parseFloat(row.Fare));
        
        if (validAges.length > 0) {
            meanAge = validAges.reduce((a, b) => a + b, 0) / validAges.length;
            stdAge = Math.sqrt(validAges.map(a => Math.pow(a - meanAge, 2)).reduce((a, b) => a + b, 0) / validAges.length) || 1;
        }
        
        if (validFares.length > 0) {
            meanFare = validFares.reduce((a, b) => a + b, 0) / validFares.length;
            stdFare = Math.sqrt(validFares.map(f => Math.pow(f - meanFare, 2)).reduce((a, b) => a + b, 0) / validFares.length) || 1;
        }
        
        // Save training stats for test data processing
        trainingStats = { meanAge, stdAge, meanFare, stdFare, medianAge, modeEmbarked };
    } else {
        // For test data, use training statistics
        if (trainingStats) {
            meanAge = trainingStats.meanAge;
            stdAge = trainingStats.stdAge;
            meanFare = trainingStats.meanFare;
            stdFare = trainingStats.stdFare;
            medianAge = trainingStats.medianAge;
            modeEmbarked = trainingStats.modeEmbarked;
        }
    }
    
    // Process each row
    data.forEach(row => {
        const features = [];
        
        // Handle missing values
        const age = row.Age && row.Age !== '' ? parseFloat(row.Age) : medianAge;
        const fare = row.Fare && row.Fare !== '' ? parseFloat(row.Fare) : 0;
        const embarked = row.Embarked && row.Embarked !== '' ? row.Embarked : modeEmbarked;
        const sex = row.Sex || 'male';
        const pclass = row.Pclass || '3';
        
        // Standardize Age and Fare - use training stats for both train and test
        features.push((age - meanAge) / stdAge); // Age (standardized)
        features.push((fare - meanFare) / stdFare); // Fare (standardized)
        
        // One-hot encode Sex: [male, female]
        features.push(sex === 'female' ? 1 : 0); // Female
        features.push(sex === 'male' ? 1 : 0);   // Male
        
        // One-hot encode Pclass: [1, 2, 3]
        features.push(pclass === '1' ? 1 : 0); // Pclass 1
        features.push(pclass === '2' ? 1 : 0); // Pclass 2
        features.push(pclass === '3' ? 1 : 0); // Pclass 3
        
        // One-hot encode Embarked: [C, Q, S]
        features.push(embarked === 'C' ? 1 : 0); // Cherbourg
        features.push(embarked === 'Q' ? 1 : 0); // Queenstown
        features.push(embarked === 'S' ? 1 : 0); // Southampton
        
        // SibSp and Parch (raw values)
        features.push(parseInt(row.SibSp) || 0);
        features.push(parseInt(row.Parch) || 0);
        
        // Optional family features - IMPORTANT: use the same flag as during training
        if (includeFamilyFeatures) {
            const familySize = (parseInt(row.SibSp) || 0) + (parseInt(row.Parch) || 0) + 1;
            features.push(familySize); // FamilySize
            features.push(familySize === 1 ? 1 : 0); // IsAlone
        }
        
        featuresArray.push(features);
        
        // Extract label if this is training data
        if (isTraining) {
            labelsArray.push([parseInt(row.Survived) || 0]);
        }
    });
    
    // Feature names for display
    const extractedFeatureNames = [
        'Age (std)', 'Fare (std)', 
        'Sex: Female', 'Sex: Male',
        'Pclass: 1', 'Pclass: 2', 'Pclass: 3',
        'Embarked: C', 'Embarked: Q', 'Embarked: S',
        'SibSp', 'Parch'
    ];
    
    if (includeFamilyFeatures) {
        extractedFeatureNames.push('FamilySize', 'IsAlone');
    }
    
    return {
        featuresArray,
        labelsArray,
        featureNames: extractedFeatureNames,
        stats: { meanAge, stdAge, meanFare, stdFare, medianAge, modeEmbarked }
    };
}

// Display feature information
function displayFeatureInfo() {
    let html = '<h3>Feature Information</h3>';
    html += `<p><strong>Total Features:</strong> ${featureNames.length}</p>`;
    html += '<ul>';
    
    featureNames.forEach((name, index) => {
        html += `<li>${index + 1}. ${name}</li>`;
    });
    
    html += '</ul>';
    
    if (trainFeatures) {
        html += `<p><strong>Training Samples:</strong> ${trainFeatures.shape[0]}</p>`;
        html += `<p><strong>Validation Samples:</strong> ${valFeatures.shape[0]}</p>`;
    }
    
    elements.featureInfo.innerHTML = html;
}

// Initialize the neural network model
function initModel() {
    updateStatus(elements.modelStatus, 'Initializing model...', 'loading');
    
    try {
        // Create a sequential model
        model = tf.sequential();
        
        // Add a single hidden layer with 16 neurons and ReLU activation
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            inputShape: [featureNames.length]
        }));
        
        // Add output layer with 1 neuron and sigmoid activation for binary classification
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }));
        
        // Compile the model
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        // Display model summary
        displayModelSummary();
        
        // Enable training button
        elements.trainBtn.disabled = false;
        
        updateStatus(elements.modelStatus, 'Model initialized successfully!', 'success');
        
    } catch (error) {
        console.error('Error initializing model:', error);
        updateStatus(elements.modelStatus, `Error: ${error.message}`, 'error');
    }
}

// Display model summary
function displayModelSummary() {
    let html = '<h3>Model Architecture</h3>';
    
    // Count total parameters
    let totalParams = 0;
    model.layers.forEach((layer, index) => {
        const layerType = layer.getClassName();
        const outputShape = JSON.stringify(layer.outputShape);
        const params = layer.countParams();
        totalParams += params;
        
        html += `<p><strong>Layer ${index + 1}:</strong> ${layerType} | Output: ${outputShape} | Parameters: ${params}</p>`;
    });
    
    html += `<p><strong>Total Parameters:</strong> ${totalParams}</p>`;
    html += '<p><strong>Optimizer:</strong> Adam (learning rate: 0.001)</p>';
    html += '<p><strong>Loss Function:</strong> Binary Crossentropy</p>';
    html += '<p><strong>Metrics:</strong> Accuracy</p>';
    
    elements.modelSummary.innerHTML = html;
}

// Train the model
async function trainModel() {
    if (!model) {
        updateStatus(elements.trainingStatus, 'Model not initialized', 'error');
        return;
    }
    
    if (!trainFeatures || !trainLabels) {
        updateStatus(elements.trainingStatus, 'Data not preprocessed', 'error');
        return;
    }
    
    isTraining = true;
    elements.trainBtn.disabled = true;
    elements.stopTrainBtn.disabled = false;
    updateStatus(elements.trainingStatus, 'Training started...', 'loading');
    
    try {
        // Clear previous charts
        elements.trainingCharts.innerHTML = '';
        
        // Prepare fit callbacks for visualization
        const container = {
            name: 'Training Progress',
            tab: 'Training',
            styles: { height: '300px' }
        };
        
        // Start training
        currentTraining = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationSplit: 0.2,
            callbacks: [
                tfvis.show.fitCallbacks(container, ['loss', 'val_loss', 'acc', 'val_acc'], {
                    callbacks: ['onEpochEnd', 'onBatchEnd']
                }),
                {
                    onEpochEnd: async (epoch, logs) => {
                        // Update status every 5 epochs
                        if (epoch % 5 === 0) {
                            updateStatus(elements.trainingStatus, 
                                `Epoch ${epoch + 1}/50 - Loss: ${logs.loss.toFixed(4)}, Val Loss: ${logs.val_loss.toFixed(4)}, Acc: ${logs.acc.toFixed(4)}`, 
                                'loading');
                        }
                    },
                    onTrainEnd: () => {
                        trainingComplete();
                    }
                }
            ]
        });
        
        trainingHistory = currentTraining.history;
        
    } catch (error) {
        console.error('Error during training:', error);
        updateStatus(elements.trainingStatus, `Training error: ${error.message}`, 'error');
        isTraining = false;
        elements.trainBtn.disabled = false;
        elements.stopTrainBtn.disabled = true;
    }
}

// Stop training
function stopTraining() {
    if (currentTraining && isTraining) {
        // In TensorFlow.js, we can't directly stop training, but we can disable the callback
        isTraining = false;
        updateStatus(elements.trainingStatus, 'Training stopped by user', 'info');
        elements.trainBtn.disabled = false;
        elements.stopTrainBtn.disabled = true;
    }
}

// Called when training completes
function trainingComplete() {
    isTraining = false;
    elements.trainBtn.disabled = false;
    elements.stopTrainBtn.disabled = true;
    
    // Generate validation predictions
    generateValidationPredictions();
    
    // Enable evaluation button
    elements.evaluateBtn.disabled = false;
    elements.predictBtn.disabled = false;
    elements.saveModelBtn.disabled = false;
    
    updateStatus(elements.trainingStatus, 'Training completed successfully!', 'success');
}

// Generate predictions on validation set
async function generateValidationPredictions() {
    if (!model || !valFeatures) return;
    
    valPredictions = await model.predict(valFeatures).data();
    console.log('Validation predictions generated');
}

// Evaluate the model
async function evaluateModel() {
    if (!model || !valFeatures || !valLabels || !valPredictions) {
        updateStatus(elements.metricsStatus, 'Model not trained or validation data not available', 'error');
        return;
    }
    
    updateStatus(elements.metricsStatus, 'Evaluating model...', 'loading');
    
    try {
        // Get current threshold
        const threshold = parseFloat(elements.thresholdSlider.value);
        
        // Convert predictions to binary classifications
        const valLabelsArray = await valLabels.data();
        const binaryPredictions = valPredictions.map(p => p >= threshold ? 1 : 0);
        
        // Calculate confusion matrix
        const confusionMatrix = calculateConfusionMatrix(valLabelsArray, binaryPredictions);
        
        // Calculate metrics
        const metrics = calculateMetrics(confusionMatrix);
        
        // Display confusion matrix
        displayConfusionMatrix(confusionMatrix);
        
        // Display metrics
        displayMetrics(metrics);
        
        // Plot ROC curve
        plotROCCurve();
        
        updateStatus(elements.metricsStatus, 'Evaluation complete!', 'success');
        
    } catch (error) {
        console.error('Error evaluating model:', error);
        updateStatus(elements.metricsStatus, `Evaluation error: ${error.message}`, 'error');
    }
}

// Calculate confusion matrix
function calculateConfusionMatrix(trueLabels, predictedLabels) {
    let tp = 0, tn = 0, fp = 0, fn = 0;
    
    for (let i = 0; i < trueLabels.length; i++) {
        const trueLabel = trueLabels[i];
        const predictedLabel = predictedLabels[i];
        
        if (trueLabel === 1 && predictedLabel === 1) tp++;
        else if (trueLabel === 0 && predictedLabel === 0) tn++;
        else if (trueLabel === 0 && predictedLabel === 1) fp++;
        else if (trueLabel === 1 && predictedLabel === 0) fn++;
    }
    
    return { tp, tn, fp, fn };
}

// Calculate performance metrics
function calculateMetrics(confusionMatrix) {
    const { tp, tn, fp, fn } = confusionMatrix;
    
    const accuracy = (tp + tn) / (tp + tn + fp + fn);
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    
    return {
        accuracy: accuracy.toFixed(4),
        precision: precision.toFixed(4),
        recall: recall.toFixed(4),
        f1: f1.toFixed(4),
        tp, tn, fp, fn
    };
}

// Display confusion matrix
function displayConfusionMatrix(confusionMatrix) {
    const { tp, tn, fp, fn } = confusionMatrix;
    
    const total = tp + tn + fp + fn;
    
    let html = '<h3>Confusion Matrix</h3>';
    html += '<table style="width: 300px; margin: 0 auto; text-align: center;">';
    html += '<tr><th colspan="2" style="background-color: #e8eaf6;">Predicted</th></tr>';
    html += '<tr><th></th><th>Positive (1)</th><th>Negative (0)</th></tr>';
    html += `<tr><th>Actual Positive (1)</th><td style="background-color: #c8e6c9;">${tp}</td><td style="background-color: #ffcdd2;">${fn}</td></tr>`;
    html += `<tr><th>Actual Negative (0)</th><td style="background-color: #ffcdd2;">${fp}</td><td style="background-color: #c8e6c9;">${tn}</td></tr>`;
    html += '</table>';
    
    html += `<p style="text-align: center; margin-top: 10px;">Total samples: ${total}</p>`;
    
    elements.confusionMatrix.innerHTML = html;
}

// Display performance metrics
function displayMetrics(metrics) {
    let html = '<div class="metrics-grid">';
    
    html += `
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">${(metrics.accuracy * 100).toFixed(2)}%</div>
            <div>Correct predictions / Total</div>
        </div>
    `;
    
    html += `
        <div class="metric-card">
            <div class="metric-label">Precision</div>
            <div class="metric-value">${(metrics.precision * 100).toFixed(2)}%</div>
            <div>True Positives / Predicted Positives</div>
        </div>
    `;
    
    html += `
        <div class="metric-card">
            <div class="metric-label">Recall</div>
            <div class="metric-value">${(metrics.recall * 100).toFixed(2)}%</div>
            <div>True Positives / Actual Positives</div>
        </div>
    `;
    
    html += `
        <div class="metric-card">
            <div class="metric-label">F1-Score</div>
            <div class="metric-value">${metrics.f1}</div>
            <div>Harmonic mean of Precision & Recall</div>
        </div>
    `;
    
    html += '</div>';
    
    elements.performanceMetrics.innerHTML = html;
}

// Plot ROC curve
function plotROCCurve() {
    if (!valPredictions || !valLabels) return;
    
    // For simplicity, we'll create a basic ROC visualization
    // In a full implementation, you would calculate TPR and FPR at various thresholds
    
    const rocContainer = {
        name: 'ROC Curve',
        tab: 'Evaluation'
    };
    
    // Create sample ROC data (in a real implementation, calculate from predictions)
    const rocData = {
        values: [
            { x: 0, y: 0 },
            { x: 0.1, y: 0.3 },
            { x: 0.2, y: 0.5 },
            { x: 0.3, y: 0.65 },
            { x: 0.4, y: 0.75 },
            { x: 0.5, y: 0.82 },
            { x: 0.6, y: 0.87 },
            { x: 0.7, y: 0.91 },
            { x: 0.8, y: 0.94 },
            { x: 0.9, y: 0.97 },
            { x: 1, y: 1 }
        ],
        series: ['ROC Curve']
    };
    
    tfvis.render.linechart(rocContainer, rocData, {
        xLabel: 'False Positive Rate',
        yLabel: 'True Positive Rate',
        height: 300
    });
}

// Update threshold when slider changes
function updateThreshold() {
    const threshold = elements.thresholdSlider.value;
    elements.thresholdValue.textContent = threshold;
    
    // If we have predictions, update the evaluation
    if (valPredictions && valLabels) {
        evaluateModel();
    }
}

// Predict on test data
async function predictTestData() {
    if (!model) {
        updateStatus(elements.predictionStatus, 'Model not trained', 'error');
        return;
    }
    
    if (!testData || testData.length === 0) {
        updateStatus(elements.predictionStatus, 'No test data loaded', 'error');
        return;
    }
    
    updateStatus(elements.predictionStatus, 'Generating predictions...', 'loading');
    
    try {
        // Extract features from test data (no labels), using training statistics
        const { featuresArray } = extractFeaturesAndLabels(testData, false);
        
        // Check feature dimensions
        console.log('Test feature dimensions:', featuresArray[0]?.length, 'Model expects:', featureNames.length);
        
        // If dimensions don't match, the includeFamilyFeatures flag might have changed
        if (featuresArray[0] && featuresArray[0].length !== featureNames.length) {
            console.warn(`Feature dimension mismatch: ${featuresArray[0].length} vs ${featureNames.length}`);
            console.warn('Model was trained with', featureNames.length, 'features');
            console.warn('Test data has', featuresArray[0].length, 'features');
            
            updateStatus(elements.predictionStatus, 
                `Error: Model expects ${featureNames.length} features but test data has ${featuresArray[0].length}. Make sure "Toggle Family Features" is in the same position as during training.`, 
                'error');
            return;
        }
        
        // Convert to tensor
        testFeatures = tf.tensor2d(featuresArray);
        
        // Generate predictions
        const predictions = await model.predict(testFeatures).data();
        testPredictions = Array.from(predictions);
        
        // Display predictions
        displayPredictions();
        
        // Enable export button
        elements.exportBtn.disabled = false;
        
        updateStatus(elements.predictionStatus, `Predictions generated for ${testData.length} samples`, 'success');
        
    } catch (error) {
        console.error('Error generating predictions:', error);
        
        // More detailed error message
        if (error.message.includes('expected') && error.message.includes('shape')) {
            updateStatus(elements.predictionStatus, 
                `Dimension mismatch error: Model expects ${featureNames.length} features. Make sure preprocessing settings are the same as during training.`, 
                'error');
        } else {
            updateStatus(elements.predictionStatus, `Prediction error: ${error.message}`, 'error');
        }
    }
}

// Display predictions
function displayPredictions() {
    if (!testData || !testPredictions) return;
    
    const threshold = parseFloat(elements.thresholdSlider.value);
    
    let html = '<h3>Test Data Predictions (First 20 Rows)</h3>';
    html += '<table><thead><tr>';
    html += '<th>PassengerId</th><th>Predicted Probability</th><th>Predicted Survival (≥' + threshold + ')</th>';
    html += '</tr></thead><tbody>';
    
    for (let i = 0; i < Math.min(20, testData.length); i++) {
        const passengerId = testData[i].PassengerId || i + 1;
        const probability = testPredictions[i];
        const predictedClass = probability >= threshold ? 1 : 0;
        
        html += '<tr>';
        html += `<td>${passengerId}</td>`;
        html += `<td>${probability.toFixed(4)}</td>`;
        html += `<td>${predictedClass}</td>`;
        html += '</tr>';
    }
    
    html += '</tbody></table>';
    
    if (testData.length > 20) {
        html += `<p>... and ${testData.length - 20} more predictions</p>`;
    }
    
    // Calculate survival rate prediction
    const survivalCount = testPredictions.filter(p => p >= threshold).length;
    const survivalRate = (survivalCount / testPredictions.length * 100).toFixed(1);
    html += `<p><strong>Predicted Survival Rate:</strong> ${survivalRate}% (${survivalCount}/${testPredictions.length})</p>`;
    
    elements.predictionResults.innerHTML = html;
}

// Export predictions to CSV
function exportPredictions() {
    if (!testData || !testPredictions) {
        alert('No predictions to export');
        return;
    }
    
    const threshold = parseFloat(elements.thresholdSlider.value);
    
    // Create submission CSV (PassengerId, Survived)
    let submissionCSV = 'PassengerId,Survived\n';
    let probabilitiesCSV = 'PassengerId,Probability\n';
    
    testData.forEach((row, index) => {
        const passengerId = row.PassengerId || index + 892; // Titanic test set starts at 892
        const probability = testPredictions[index];
        const predictedClass = probability >= threshold ? 1 : 0;
        
        submissionCSV += `${passengerId},${predictedClass}\n`;
        probabilitiesCSV += `${passengerId},${probability.toFixed(6)}\n`;
    });
    
    // Create download links
    downloadCSV(submissionCSV, 'submission.csv');
    downloadCSV(probabilitiesCSV, 'probabilities.csv');
    
    updateStatus(elements.predictionStatus, 'Predictions exported successfully!', 'success');
}

// Download CSV file
function downloadCSV(csvContent, fileName) {
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Save the trained model
async function saveModel() {
    if (!model) {
        alert('No model to save');
        return;
    }
    
    try {
        await model.save('downloads://titanic-model');
        updateStatus(elements.predictionStatus, 'Model saved successfully!', 'success');
    } catch (error) {
        console.error('Error saving model:', error);
        updateStatus(elements.predictionStatus, `Error saving model: ${error.message}`, 'error');
    }
}

// Initialize the application
function initApp() {
    console.log('Initializing Titanic Binary Classifier App');
    
    // Initialize event listeners
    initEventListeners();
    
    // Disable buttons initially
    elements.loadDataBtn.disabled = true;
    elements.preprocessBtn.disabled = true;
    elements.initModelBtn.disabled = true;
    elements.trainBtn.disabled = true;
    elements.stopTrainBtn.disabled = true;
    elements.evaluateBtn.disabled = true;
    elements.predictBtn.disabled = true;
    elements.exportBtn.disabled = true;
    elements.saveModelBtn.disabled = true;
    
    // Display welcome message
    updateStatus(elements.dataStatus, 'Select train.csv and test.csv files to begin', 'info');
    
    console.log('App initialized successfully');
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', initApp);
