import React, { useEffect, useRef } from 'react';
import './ParticleEfffect.css';

const ParticleCanvas = () => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let particles = [];

    // Function to create a particle
    function Particle(x, y) {
      this.x = x;
      this.y = y;
      this.size = Math.random() * 0.4 * 5 + 0.85; // 15% of the size
      this.speedX = Math.random() * 3 - 1.5;
      this.speedY = Math.random() * 3 - 1.5;
    }

    // Function to draw particles and connect them with lines
    function drawParticles() {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (let i = 0; i < particles.length; i++) {
        ctx.fillStyle = '#000000'; // Black color particles
        ctx.beginPath();
        ctx.arc(particles[i].x, particles[i].y, particles[i].size, 0, Math.PI * 2);
        ctx.fill();

        particles[i].x += particles[i].speedX;
        particles[i].y += particles[i].speedY;

        // Wrap particles around the screen
        if (particles[i].x > canvas.width) particles[i].x = 0;
        if (particles[i].x < 0) particles[i].x = canvas.width;
        if (particles[i].y > canvas.height) particles[i].y = 0;
        if (particles[i].y < 0) particles[i].y = canvas.height;

        // Draw lines between neighboring particles
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          const opacity = 1 - distance / 100; // Opacity based on distance

          if (opacity > 0) {
            ctx.strokeStyle = `rgba(0, 0, 0, ${opacity})`; // Set line opacity
            ctx.lineWidth = 0.5; // Set line thickness
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }
        }
      }

      requestAnimationFrame(drawParticles);
    }

    // Initialize particles
    for (let i = 0; i < 120; i++) {
      particles.push(new Particle(Math.random() * canvas.width, Math.random() * canvas.height));
    }

    drawParticles();

    // Cleanup on unmount
    return () => {
      particles = [];
    };
  }, []);

  return <canvas ref={canvasRef} style={{ position: 'absolute', zIndex: -1, top: 0, left: 0, width: '100%', height: '100%' }} />;
};

export default ParticleCanvas;
