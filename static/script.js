document.addEventListener('DOMContentLoaded', function() {
    const frontInput = document.querySelector('input[name="front"]');
    const backInput = document.querySelector('input[name="back"]');
    const frontPreview = document.createElement('img');
    const backPreview = document.createElement('img');

    frontPreview.classList.add('image-preview');
    backPreview.classList.add('image-preview');

    frontInput.addEventListener('change', function() {
        if (frontInput.files && frontInput.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                frontPreview.src = e.target.result;
                frontPreview.alt = 'Front Image Preview'; // Added alt text for accessibility
                document.querySelector('.container').appendChild(frontPreview);
            }
            reader.readAsDataURL(frontInput.files[0]);
        }
    });

    backInput.addEventListener('change', function() {
        if (backInput.files && backInput.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                backPreview.src = e.target.result;
                backPreview.alt = 'Back Image Preview'; // Added alt text for accessibility
                document.querySelector('.container').appendChild(backPreview);
            }
            reader.readAsDataURL(backInput.files[0]);
        }
    });
});
