import React from 'react';
import './dashboard.css';

function Dashboard() {
  return (
    <div className="dashboard">
      <header>
        <h1 className='head'>Contact Us</h1>
      </header>

      <section>
        <p className='contact-content'>If you have any questions, feedback, or inquiries, please feel free to contact us using the information below:</p>
        <address>
          Email: info@college.edu<br />
          Phone: (123) 456-7890
        </address>
      </section>

      <footer>
        <p className='contact-content'>We look forward to hearing from you!</p>
      </footer>
    </div>
  );
}

export default Dashboard;
