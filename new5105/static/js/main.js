// Tab switching functionality
function openTab(evt, tabId) {
    const tabContents = document.getElementsByClassName('tab-content');
    for (let i = 0; i < tabContents.length; i++) {
        tabContents[i].classList.remove('active');
    }

    const tabButtons = document.getElementsByClassName('tab-button');
    for (let i = 0; i < tabButtons.length; i++) {
        tabButtons[i].classList.remove('active');
    }

    document.getElementById(tabId).classList.add('active');
    evt.currentTarget.classList.add('active');
}

// Image preview functionality
function setupImagePreview(inputId, previewId) {
    const input = document.getElementById(inputId);
    const preview = document.getElementById(previewId);

    input.addEventListener('change', function () {
        if (this.files && this.files[0]) {
            const reader = new FileReader();

            reader.onload = function (e) {
                preview.src = e.target.result;
                preview.style.display = 'block';
            };

            reader.readAsDataURL(this.files[0]);
        }
    });
}

// Set up image previews
setupImagePreview('encode-image', 'encode-preview');
setupImagePreview('decode-image', 'decode-preview');

// Encode functionality
document.getElementById('encode-button').addEventListener('click', function () {
    const imageInput = document.getElementById('encode-image');
    const message = document.getElementById('secret-message').value;
    const resultDiv = document.getElementById('encode-result');
    const errorDiv = document.getElementById('encode-error');

    // Hide previous results/errors
    resultDiv.style.display = 'none';
    errorDiv.style.display = 'none';

    // Validate input
    if (!imageInput.files || imageInput.files.length === 0) {
        errorDiv.textContent = 'Please select an image to encode.';
        errorDiv.style.display = 'block';
        return;
    }

    if (!message) {
        errorDiv.textContent = 'Please enter a message to encode.';
        errorDiv.style.display = 'block';
        return;
    }

    // Create form data for submission
    const formData = new FormData();
    formData.append('image', imageInput.files[0]);
    formData.append('message', message);

    // Show loading state
    this.textContent = 'Encoding...';
    this.disabled = true;

    // Send request to encode
    fetch('/encode', {
        method: 'POST',
        body: formData
    })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Failed to encode image');
                });
            }
            return response.blob();
        })
        .then(blob => {
            // Create object URL for the encoded image
            const objectURL = URL.createObjectURL(blob);

            // Display the encoded image
            const encodedImage = document.getElementById('encoded-image');
            encodedImage.src = objectURL;

            // Set up download button
            const downloadButton = document.getElementById('download-button');
            downloadButton.onclick = function () {
                const a = document.createElement('a');
                a.href = objectURL;
                a.download = 'encoded-image.png';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            };

            // Show the result
            resultDiv.style.display = 'block';
        })
        .catch(error => {
            errorDiv.textContent = error.message;
            errorDiv.style.display = 'block';
        })
        .finally(() => {
            // Reset button state
            this.textContent = 'Encode Image';
            this.disabled = false;
        });
});

// Decode functionality
document.getElementById('decode-button').addEventListener('click', function () {
    const imageInput = document.getElementById('decode-image');
    const resultDiv = document.getElementById('decode-result');
    const messageBox = document.getElementById('decoded-message');
    const errorDiv = document.getElementById('decode-error');

    // Hide previous results/errors
    resultDiv.style.display = 'none';
    errorDiv.style.display = 'none';

    // Validate input
    if (!imageInput.files || imageInput.files.length === 0) {
        errorDiv.textContent = 'Please select an image to decode.';
        errorDiv.style.display = 'block';
        return;
    }

    // Create form data for submission
    const formData = new FormData();
    formData.append('image', imageInput.files[0]);

    // Show loading state
    this.textContent = 'Decoding...';
    this.disabled = true;

    // Send request to decode
    fetch('/decode', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }

            // Display the decoded message
            messageBox.textContent = data.message;
            resultDiv.style.display = 'block';
        })
        .catch(error => {
            errorDiv.textContent = error.message;
            errorDiv.style.display = 'block';
        })
        .finally(() => {
            // Reset button state
            this.textContent = 'Decode Image';
            this.disabled = false;
        });
}); 