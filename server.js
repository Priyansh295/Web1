// server.js
const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');
const cors = require('cors');
const app = express();
const PORT = process.env.PORT || 5000;

app.use(bodyParser.json());

app.use(cors()); // Enable CORS for all routes

// ... (your existing code)

// Connect to MongoDB
mongoose.connect('mongodb://127.0.0.1:27017/student', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});
// Define a mongoose schema for users
const userSchema = new mongoose.Schema({
  username: String,
  password: String,
});

const User = mongoose.model('User', userSchema);

// Seed the database with mock data (run this only once)
const seedDatabase = async () => {
  try {
    await User.insertMany([
      { username: 'Priyansh', password: 'priyansh123' },
      { username: 'Yash', password: 'yash123' },
      { username: 'Vinnay', password: 'vinnay123' },
    ]);
    console.log('Database seeded successfully');
  } catch (error) {
    console.error('Error seeding the database:', error);
  }
};

seedDatabase();

// API endpoint for user authentication
app.post('/login', async (req, res) => {
  const { username, password } = req.body;

  try {
    // Check if the entered username and password match any in the database
    const user = await User.findOne({ username, password });

    if (user) {
      // Authentication successful
      res.json({ success: true });
    } else {
      // Authentication failed
      res.json({ success: false, message: 'Invalid username or password' });
    }
  } catch (error) {
    console.error(error);
    res.status(500).json({ success: false, message: 'Server error' });
  }
});

app.listen(PORT, () => {
  console.log('Server is running on port ${PORT}');
});