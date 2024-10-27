document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('uploadForm');
    const frontInput = document.getElementById('front');
    const backInput = document.getElementById('back');
    const errorMessage = document.getElementById('errorMessage');
    const loadingSpinner = document.getElementById('loadingSpinner');

    function validateFileType(fileInput) {
        const file = fileInput.files[0];
        if (file && !file.type.startsWith('image/')) {
            return false;
        }
        return true;
    }

    function displayError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
    }

    form.addEventListener('submit', (event) => {
        errorMessage.style.display = 'none';
        
        if (!frontInput.files.length || !backInput.files.length) {
            displayError('Please upload both front and back images.');
            event.preventDefault();
            return;
        }
        
        if (!validateFileType(frontInput) || !validateFileType(backInput)) {
            displayError('Only image files are allowed.');
            event.preventDefault();
            return;
        }
        
        // Show loading spinner and proceed with form submission
        loadingSpinner.style.display = 'flex';
    });

    // Hide spinner when page loads (if returning to this page with form data)
    window.addEventListener('load', () => {
        loadingSpinner.style.display = 'none';
    });
});

function previewImage(event, previewId) {
    const input = event.target;
    const preview = document.getElementById(previewId);

    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = 'block'; // Show the preview
        };
        reader.readAsDataURL(input.files[0]);
    } else {
        preview.src = '';
        preview.style.display = 'none'; // Hide the preview if no file is selected
    }
}
