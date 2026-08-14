// frontend/src/App.test.js
//
// Replaces the default Create React App placeholder test (which checked
// for "learn react" — leftover boilerplate text that never existed in
// this actual app). These test the real login screen instead, which is
// what App renders by default when there's no auth token in localStorage.

import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the login screen when not authenticated', () => {
  render(<App />);
  const heading = screen.getByText(/log in to your terminal/i);
  expect(heading).toBeInTheDocument();
});

test('shows email and password fields on the login screen', () => {
  render(<App />);
  expect(screen.getByPlaceholderText(/you@example.com/i)).toBeInTheDocument();
  expect(screen.getByPlaceholderText(/at least 8 characters/i)).toBeInTheDocument();
});

test("shows a link to switch to the registration screen", () => {
  render(<App />);
  expect(screen.getByText(/don't have an account\? register/i)).toBeInTheDocument();
});