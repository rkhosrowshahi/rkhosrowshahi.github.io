import React, { useState, useEffect } from 'react';
// import Typewriter from 'typewriter-effect';
import Fade from 'react-reveal';
import endpoints from '../constants/endpoints';
import Social from './Social';
import FallbackSpinner from './FallbackSpinner';

const styles = {
  imgStyle: {
    width: 300,
    height: 300,
  },
  nameStyle: {
    fontSize: '5em',
  },
  inlineChild: {
    display: 'inline-block',
  },
  mainContainer: {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
  },
};

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
    <Fade>
      <div style={styles.mainContainer}>
        <img
          src={data?.personal?.source}
          className="d-inline-block align-top"
          alt="main logo"
          style={
            data?.personal?.height && data?.personal?.width
              ? { height: data?.personal?.height, width: data?.personal?.width }
              : styles.logoStyle
          }
        />
        <h1 style={styles.nameStyle}>{data?.name}</h1>
        <div style={{ flexDirection: 'row' }}>
          <h2>{data?.roles}</h2>
          {/* <h2 style={styles.inlineChild}>I&apos;m&nbsp;</h2> */}
          {/* <Typewriter
            options={{
              loop: false,
              autoStart: true,
              strings: data?.roles,
            }} */}
          {/* /> */}
        </div>
        <Social />
      </div>
    </Fade>
  ) : <FallbackSpinner />;
}

export default Home;
