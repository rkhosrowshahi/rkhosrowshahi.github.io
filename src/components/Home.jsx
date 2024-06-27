import React, { useState, useEffect } from 'react';
// import Typewriter from 'typewriter-effect';
import endpoints from '../constants/endpoints';
import Social from './Social';
import FallbackSpinner from './FallbackSpinner';
import '../css/home.css';

function Home() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch(endpoints.home, {
      method: 'GET',
    })
      .then((res) => res.json())
      .then((res) => setData(res))
      .catch((err) => err);
  }, []);

  return data ? (
    <div className="main_container">
      <h1 className="name_style">{data?.name}</h1>
      <img
        src={data?.personal?.source}
        alt=""
        className="rounded"
      />
      <div className="inline_child">
        {data?.roles.map((item) => (
          <p key={Math.random()}>{item}</p>
        ))}
      </div>
      <Social />
    </div>
  ) : <FallbackSpinner />;
}

export default Home;
