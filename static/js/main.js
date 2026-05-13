// IntelliCloud Attendance System - Main JS
console.log("IntelliCloud Attendance System v1.0 loaded");

// Utility: format date to YYYY-MM-DD
function formatDate(date) {
  return date.toISOString().split('T')[0];
}

// Set today's date as default in date inputs
document.querySelectorAll('input[type="date"]').forEach(el => {
  el.value = formatDate(new Date());
});
