document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('uploadForm');
    const frontInput = document.getElementById('front');
    const backInput = document.getElementById('back');
    const errorMessage = document.getElementById('errorMessage');
    const heroButton = document.querySelector('.hero-button');

    function validateFileType(fileInput) {
        const file = fileInput.files[0];
        return file && file.type.startsWith('image/');
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
        }
    });

    heroButton.addEventListener('click', (event) => {
        event.preventDefault();
        document.querySelector(heroButton.getAttribute('href')).scrollIntoView({ behavior: 'smooth' });
    });
});
