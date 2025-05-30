// Global variables for attendance tracking
let isProcessing = false;
let blinkDetected = false;
let faceDetected = false;

// Utility function to show notifications
function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.querySelector('.container').insertBefore(alertDiv, document.querySelector('.container').firstChild);
    
    // Auto dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// Initialize Bootstrap tooltips
document.addEventListener('DOMContentLoaded', function() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// Handle file input preview
const imageInput = document.querySelector('input[type="file"]');
if (imageInput) {
    imageInput.addEventListener('change', function(e) {
        if (this.files && this.files[0]) {
            const reader = new FileReader();
            const preview = document.querySelector('#preview');
            
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.style.display = 'block';
            }
            
            reader.readAsDataURL(this.files[0]);
        }
    });
}

// Handle form submissions with AJAX
document.querySelectorAll('form').forEach(form => {
    form.addEventListener('submit', function(e) {
        if (this.getAttribute('data-ajax') === 'true') {
            e.preventDefault();
            
            const formData = new FormData(this);
            const submitButton = this.querySelector('button[type="submit"]');
            submitButton.disabled = true;
            
            fetch(this.action, {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showNotification(data.message, 'success');
                    if (data.redirect) {
                        window.location.href = data.redirect;
                    }
                } else {
                    showNotification(data.message || 'An error occurred', 'danger');
                }
            })
            .catch(error => {
                showNotification('An error occurred while processing your request', 'danger');
                console.error('Error:', error);
            })
            .finally(() => {
                submitButton.disabled = false;
            });
        }
    });
});

// Handle dynamic table sorting
document.querySelectorAll('th[data-sort]').forEach(header => {
    header.addEventListener('click', function() {
        const table = this.closest('table');
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const index = Array.from(this.parentElement.children).indexOf(this);
        const ascending = this.classList.toggle('sort-asc');
        
        rows.sort((a, b) => {
            const aValue = a.children[index].textContent;
            const bValue = b.children[index].textContent;
            return ascending ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
        });
        
        rows.forEach(row => tbody.appendChild(row));
    });
});

// Handle table search/filter
const tableFilter = document.querySelector('#tableFilter');
if (tableFilter) {
    tableFilter.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        const table = document.querySelector('table');
        const rows = table.querySelectorAll('tbody tr');
        
        rows.forEach(row => {
            const text = row.textContent.toLowerCase();
            row.style.display = text.includes(searchTerm) ? '' : 'none';
        });
    });
}

// Export table to Excel
function exportTableToExcel(tableId, filename = 'report') {
    const table = document.getElementById(tableId);
    const wb = XLSX.utils.table_to_book(table);
    XLSX.writeFile(wb, `${filename}.xlsx`);
}

// 