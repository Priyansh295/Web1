import React, { useState , useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './style.css';

function Login({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (event) => {
    event.preventDefault();

    try {
      // Send a request to the server to check user credentials
      const response = await fetch('http://localhost:5000/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      if (response.ok) {
        // Login successful
        onLogin();
        navigate('/dashboard');
      } else {
        // Handle login failure, e.g., display an error message
        console.log('Invalid username or password');
      }
    } catch (error) {
      // Handle network or server error
      console.error('Error during login:', error);
    }

  };
  useEffect(() => {
    const signInBtnLink = document.querySelector('.signInBtn-link');
    const signUpBtnLink = document.querySelector('.signUpBtn-link');
    const wrapper = document.querySelector('.wrapper');

    const toggleActiveClass = () => {
      wrapper.classList.toggle('active');
    };

    signUpBtnLink.addEventListener('click', toggleActiveClass);
    signInBtnLink.addEventListener('click', toggleActiveClass);

    return () => {
      // Cleanup: Remove event listeners when the component unmounts
      signUpBtnLink.removeEventListener('click', toggleActiveClass);
      signInBtnLink.removeEventListener('click', toggleActiveClass);
    };
  }, []);
  return (
    <div className="container">
        <h1 className='welcome'>Welcome To <br/> Student Portal</h1>
    <div className="wrapper login-page">
      <div className="form-wrapper sign-in">
        <form onSubmit={handleSubmit}>
          <h2 className='head2'>Login</h2>
          <div className="input-group">
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
            <label htmlFor="">Username</label>
          </div>
          <div className="input-group">
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            <label htmlFor="">Password</label>
          </div>
          <div className="remember">
            <label>
              <input type="checkbox" /> Remember me
            </label>
          </div>
          <button className="buttons" type="submit">
            Login
          </button>
          <div className="signUp-link">
            <p>
              Don't have an account?{' '}
              <a href="#" className="signUpBtn-link">
                Sign Up
              </a>
            </p>
          </div>
        </form>
      </div>
      <div className="form-wrapper sign-up">
        <form action="">
          <h2 className='head2'>Sign Up</h2>
          <div className="input-group">
            <input type="text" required />
            <label htmlFor="">Username</label>
          </div>
          <div className="input-group">
            <input type="email" required />
            <label htmlFor="">Email</label>
          </div>
          <div className="input-group">
            <input type="password" required />
            <label htmlFor="">Password</label>
          </div>
          <div className="remember">
            <label>
              <input type="checkbox" /> I agree to the terms & conditions
            </label>
          </div>
          <button className="buttons" type="submit">
            Sign Up
          </button>

          <div className="signUp-link">
            <p>
              Already have an account?{' '}
              <a href="#" className="signInBtn-link">
                Sign In
              </a>
            </p>
          </div>
        </form>
      </div>
    </div>
    </div>
  );
}

export default Login;
