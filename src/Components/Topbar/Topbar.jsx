import React from "react";
import { Box, IconButton } from "@mui/material";
import NotificationsOutlinedIcon from "@mui/icons-material/NotificationsOutlined";
import SettingsOutlinedIcon from "@mui/icons-material/SettingsOutlined";
import PersonOutlinedIcon from "@mui/icons-material/PersonOutlined";

const Topbar = ({ onLogout }) => {
  return (
    <Box
      style={{
        background: 'linear-gradient(135deg, #512222 0%, #512222 98%, #1a1b1a 2%, #1a1b1a 100%)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '10px 20px', // Add some padding for better layout
      }}
    >
      <Box flex="1" display="flex" justifyContent="center">
        {/* Lab Manual Portal heading with white color */}
        <h1 style={{ color: '#ffffff', fontSize: '50px', fontFamily: 'ui-sans-serif, Roboto, Helvetica Neue, Arial, sans-serif' }}>
          Lab Manual Portal
        </h1>
      </Box>
      {/* ICONS */}
      <Box display="flex">
        <IconButton sx={{ color: '#ffffff' }}>
          <NotificationsOutlinedIcon />
        </IconButton>
        <IconButton sx={{ color: '#ffffff' }}>
          <SettingsOutlinedIcon />
        </IconButton>
        <IconButton sx={{ color: '#ffffff' }}>
          <PersonOutlinedIcon />
        </IconButton>
      </Box>
    </Box>
  );
};

export default Topbar;
