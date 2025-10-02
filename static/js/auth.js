// Authentication JavaScript

// Form validation rules
const validationRules = {
    email: {
        required: true,
        pattern: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
        message: 'Please enter a valid email address'
    },
    password: {
        required: true,
        minLength: 6,
        message: 'Password must be at least 6 characters long'
    },
    name: {
        required: true,
        minLength: 2,
        message: 'Name must be at least 2 characters long'
    }
};

// Initialize authentication page
function initAuth() {
    setupFormListeners();
    checkAlreadyLoggedIn();
}

// Check if user is already logged in
function checkAlreadyLoggedIn() {
    const userData = sessionStorage.getItem('currentUser');
    if (userData) {
        window.location.href = 'dashboard.html';
    }
}

// Setup form event listeners
function setupFormListeners() {
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    
    if (loginForm) {
        loginForm.addEventListener('submit', handleLogin);
    }
    
    if (registerForm) {
        registerForm.addEventListener('submit', handleRegister);
    }
    
    // Add real-time validation
    setupRealTimeValidation();
}

// Handle login form submission
async function handleLogin(e) {
    e.preventDefault();
    
    const form = e.target;
    const formData = new FormData(form);
    const loginData = {
        email: formData.get('email'),
        password: formData.get('password')
    };
    
    // Validate form
    if (!validateLoginForm(loginData)) {
        return;
    }
    
    // Show loading state
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    submitBtn.textContent = 'Signing In...';
    submitBtn.disabled = true;
    
    try {
        // Send API request to backend
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(loginData)
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            // Store user data
            sessionStorage.setItem('currentUser', JSON.stringify(data.user));
            
            // Show success message and redirect
            showAlert('Login successful! Redirecting...', 'success');
            setTimeout(() => {
                window.location.href = '/dashboard';
            }, 1500);
        } else {
            showAlert(data.error || 'Login failed', 'error');
        }
    } catch (error) {
        console.error('Login error:', error);
        showAlert('An error occurred during login. Please try again.', 'error');
    } finally {
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
    }
}

// Handle register form submission
async function handleRegister(e) {
    e.preventDefault();
    
    const form = e.target;
    const formData = new FormData(form);
    const registerData = {
        name: formData.get('name'),
        email: formData.get('email'),
        password: formData.get('password')
    };
    
    // Validate form
    if (!validateRegisterForm(registerData)) {
        return;
    }
    
    // Show loading state
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    submitBtn.textContent = 'Creating Account...';
    submitBtn.disabled = true;
    
    try {
        // Send API request to backend
        const response = await fetch('/api/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(registerData)
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            showAlert('Account created successfully! Please sign in.', 'success');
            
            // Clear form
            form.reset();
            
            // Redirect to login after 2 seconds
            setTimeout(() => {
                window.location.href = '/login';
            }, 2000);
        } else {
            showAlert(data.error || 'Registration failed', 'error');
        }
    } catch (error) {
        console.error('Registration error:', error);
        showAlert('An error occurred during registration. Please try again.', 'error');
    } finally {
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
    }
}

// Validate login form
function validateLoginForm(data) {
    const errors = [];
    
    if (!validateField('email', data.email)) {
        errors.push('Please enter a valid email address');
    }
    
    if (!validateField('password', data.password)) {
        errors.push('Password is required');
    }
    
    if (errors.length > 0) {
        showAlert(errors.join('. '), 'error');
        return false;
    }
    
    return true;
}

// Validate register form
function validateRegisterForm(data) {
    const errors = [];
    
    if (!validateField('name', data.name)) {
        errors.push(validationRules.name.message);
    }
    
    if (!validateField('email', data.email)) {
        errors.push(validationRules.email.message);
    }
    
    if (!validateField('password', data.password)) {
        errors.push(validationRules.password.message);
    }
    
    if (errors.length > 0) {
        showAlert(errors.join('. '), 'error');
        return false;
    }
    
    return true;
}

// Validate individual field
function validateField(fieldName, value) {
    const rule = validationRules[fieldName];
    if (!rule) return true;
    
    // Check required
    if (rule.required && (!value || value.trim() === '')) {
        return false;
    }
    
    // Check minimum length
    if (rule.minLength && value.length < rule.minLength) {
        return false;
    }
    
    // Check pattern
    if (rule.pattern && !rule.pattern.test(value)) {
        return false;
    }
    
    return true;
}

// Setup real-time validation
function setupRealTimeValidation() {
    const inputs = document.querySelectorAll('.form-control');
    
    inputs.forEach(input => {
        input.addEventListener('blur', function() {
            validateInputField(this);
        });
        
        input.addEventListener('input', function() {
            clearInputError(this);
        });
    });
}

// Validate individual input field
function validateInputField(input) {
    const fieldName = input.name;
    const value = input.value;
    
    if (validateField(fieldName, value)) {
        input.classList.remove('is-invalid');
        input.classList.add('is-valid');
    } else {
        input.classList.remove('is-valid');
        input.classList.add('is-invalid');
    }
}

// Clear input error styling
function clearInputError(input) {
    input.classList.remove('is-invalid', 'is-valid');
}

// Simulate user authentication (replace with actual API call)
async function authenticateUser(credentials) {
    return new Promise((resolve) => {
        setTimeout(() => {
            // Simple simulation - in real app, this would be an API call
            if (credentials.email === 'demo@example.com' && credentials.password === 'demo123') {
                resolve({
                    success: true,
                    user: {
                        id: 1,
                        name: 'Demo User',
                        email: credentials.email
                    }
                });
            } else {
                // For demo purposes, accept any email/password combination
                resolve({
                    success: true,
                    user: {
                        id: Date.now(),
                        name: credentials.email.split('@')[0],
                        email: credentials.email
                    }
                });
            }
        }, 1000);
    });
}

// Simulate user registration (replace with actual API call)
async function registerUser(userData) {
    return new Promise((resolve) => {
        setTimeout(() => {
            // Simple simulation - in real app, this would be an API call
            resolve({
                success: true,
                message: 'Account created successfully',
                user: {
                    id: Date.now(),
                    name: userData.name,
                    email: userData.email
                }
            });
        }, 1000);
    });
}

// Show alert messages
function showAlert(message, type = 'info') {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.alert');
    existingAlerts.forEach(alert => alert.remove());
    
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.innerHTML = `
        <span>${message}</span>
        <button type="button" class="alert-close" onclick="this.parentElement.remove()">&times;</button>
    `;
    
    // Insert at the top of the form container
    const formContainer = document.querySelector('.auth-right') || document.querySelector('.container');
    if (formContainer) {
        formContainer.insertBefore(alertDiv, formContainer.firstChild);
        
        // Remove alert after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.parentNode.removeChild(alertDiv);
            }
        }, 5000);
    }
}

// Password strength checker
function checkPasswordStrength(password) {
    let strength = 0;
    const feedback = [];
    
    if (password.length >= 8) {
        strength += 1;
    } else {
        feedback.push('Use at least 8 characters');
    }
    
    if (/[a-z]/.test(password)) {
        strength += 1;
    } else {
        feedback.push('Include lowercase letters');
    }
    
    if (/[A-Z]/.test(password)) {
        strength += 1;
    } else {
        feedback.push('Include uppercase letters');
    }
    
    if (/[0-9]/.test(password)) {
        strength += 1;
    } else {
        feedback.push('Include numbers');
    }
    
    if (/[^A-Za-z0-9]/.test(password)) {
        strength += 1;
    } else {
        feedback.push('Include special characters');
    }
    
    return {
        strength,
        feedback,
        level: strength <= 2 ? 'weak' : strength <= 3 ? 'medium' : 'strong'
    };
}

// Toggle password visibility
function togglePassword(inputId) {
    const input = document.getElementById(inputId);
    const toggleBtn = input.nextElementSibling;
    
    if (input.type === 'password') {
        input.type = 'text';
        toggleBtn.textContent = '👁️';
    } else {
        input.type = 'password';
        toggleBtn.textContent = '👁️‍🗨️';
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initAuth();
});

// Export functions for global access
window.AuthApp = {
    initAuth,
    handleLogin,
    handleRegister,
    showAlert,
    togglePassword,
    checkPasswordStrength
};