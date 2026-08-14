// backend/tests/auth.test.js
//
// These specifically test validation logic that runs BEFORE any database
// query — register/login return a 400 for bad input without ever hitting
// Postgres, so this suite runs without a live DB connection. Testing the
// full success path (real registration, real login) would need either a
// disposable test database or a mocked `pg` Pool — a natural next step,
// not required to demonstrate the pattern here.
//
// Run with: npx jest tests/auth.test.js

const request = require('supertest');
const app = require('../server');

describe('POST /api/auth/register — validation', () => {
  test('rejects missing email', async () => {
    const res = await request(app)
      .post('/api/auth/register')
      .send({ password: 'somepassword123' });
    expect(res.status).toBe(400);
    expect(res.body.success).toBe(false);
  });

  test('rejects missing password', async () => {
    const res = await request(app)
      .post('/api/auth/register')
      .send({ email: 'test@example.com' });
    expect(res.status).toBe(400);
  });

  test('rejects a password under 8 characters', async () => {
    const res = await request(app)
      .post('/api/auth/register')
      .send({ email: 'test@example.com', password: 'short' });
    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/8 characters/i);
  });
});

describe('POST /api/auth/login — validation', () => {
  test('rejects missing credentials', async () => {
    const res = await request(app).post('/api/auth/login').send({});
    expect(res.status).toBe(400);
    expect(res.body.success).toBe(false);
  });
});

describe('protected routes without a token', () => {
  test('GET /api/portfolio returns 401 with no Authorization header', async () => {
    const res = await request(app).get('/api/portfolio');
    expect(res.status).toBe(401);
  });

  test('POST /api/trade/execute returns 401 with no Authorization header', async () => {
    const res = await request(app)
      .post('/api/trade/execute')
      .send({ symbol: 'AAPL', action: 'BUY', quantity: 1 });
    expect(res.status).toBe(401);
  });
});

describe('GET /unknown-route', () => {
  test('returns 404 with a list of available endpoints', async () => {
    const res = await request(app).get('/api/this-does-not-exist');
    expect(res.status).toBe(404);
    expect(Array.isArray(res.body.availableEndpoints)).toBe(true);
  });
});