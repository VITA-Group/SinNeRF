/* eslint-disable max-classes-per-file */
/* eslint-disable react/no-multi-comp */

import { createMedia } from "@artsy/fresnel";
import PropTypes from "prop-types";
import "semantic-ui-css/semantic.css";
import React, { Component } from "react";
import {
  Button,
  Container,
  Divider,
  Grid,
  Header,
  Icon,
  Image,
  List,
  Menu,
  Segment,
  Sidebar,
  Visibility,
} from "semantic-ui-react";

import Abstract from "./Abstract.js";
import Results from "./Results.js";
import Comparison from "./Comparison.js";
import overview from "./images/SinNeRF.drawio.jpg";
// import video_website from "./video/video-website.mp4";

import subsection_style from "./style.js";

const { MediaContextProvider, Media } = createMedia({
  breakpoints: {
    mobile: 0,
    tablet: 768,
    computer: 1024,
  },
});

/* Heads up!
 * HomepageHeading uses inline styling, however it's not the best practice. Use CSS or styled
 * components for such things.
 */
const HomepageHeading = ({ mobile }) => (
  <Container fluid>
    {/* <Header
      as="h1"
      content="SinNeRF"
      // inverted
      textAlign="center"
      style={{
        fontSize: mobile ? "2em" : "4em",
        fontWeight: "normal",
        marginBottom: 0,
        marginTop: mobile ? "0.5em" : "1em",
      }}
    /> */}
    <Header
      as="h1"
      // inverted
      textAlign="center"
      style={{
        fontSize: mobile ? "1.5em" : "1.7em",
        fontWeight: "normal",
        marginTop: mobile ? "0.5em" : "1em",
      }}
    >
      <div>
        SinNeRF: Training Neural Radiance Fields on Complex Scenes <br></br>
        from a <b>Single</b> Image
      </div>
    </Header>
  </Container>
);

HomepageHeading.propTypes = {
  mobile: PropTypes.bool,
};

/* Heads up!
 * Neither Semantic UI nor Semantic UI React offer a responsive navbar, however, it can be implemented easily.
 * It can be more complicated, but you can create really flexible markup.
 */
class DesktopContainer extends Component {
  state = {};

  hideFixedMenu = () => this.setState({ fixed: false });
  showFixedMenu = () => this.setState({ fixed: true });

  render() {
    const { children } = this.props;
    const { fixed } = this.state;

    return (
      <Media greaterThan="mobile">
        <HomepageHeading />
        {children}
      </Media>
    );
  }
}

DesktopContainer.propTypes = {
  children: PropTypes.node,
};

class MobileContainer extends Component {
  state = {};

  handleSidebarHide = () => this.setState({ sidebarOpened: false });

  handleToggle = () => this.setState({ sidebarOpened: true });

  render() {
    const { children } = this.props;
    const { sidebarOpened } = this.state;

    return (
      <Media as={Sidebar.Pushable} at="mobile">
        <HomepageHeading mobile />

        {children}
      </Media>
    );
  }
}

MobileContainer.propTypes = {
  children: PropTypes.node,
};

const ResponsiveContainer = ({ children }) => (
  /* Heads up!
   * For large applications it may not be best option to put all page into these containers at
   * they will be rendered twice for SSR.
   */
  <MediaContextProvider>
    <DesktopContainer>{children}</DesktopContainer>
    <MobileContainer>{children}</MobileContainer>
  </MediaContextProvider>
);

ResponsiveContainer.propTypes = {
  children: PropTypes.node,
};

const HomepageLayout = ({ mobile }) => (
  <ResponsiveContainer>
    <div>
      <Container text>
        <Grid
          container
          stackable
          verticalAlign="middle"
          columns={4}
          textAlign="center"
          style={{
            fontSize: "1em",
            fontWeight: "normal",
            marginTop: "0.5em",
            marginBottom: "0.5em",
          }}
        >
          {/* <Grid.Row> */}
          {/* <Grid.Column> */}
          <a href="https://ir1d.github.io/" target="_blank" rel="noreferrer">
            Dejia Xu<sup>1</sup>*
          </a>
          {/* </Grid.Column> */}
          {/* <Grid.Column> */}
          <a href="https://yifanjiang.net/" target="_blank" rel="noreferrer">
            Yifan Jiang<sup>1</sup>*
          </a>
          {/* </Grid.Column> */}
          {/* <Grid.Column> */}
          <a
            href="https://peihaowang.github.io/"
            target="_blank"
            rel="noreferrer"
          >
            Peihao Wang<sup>1</sup>
          </a>
          {/* </Grid.Column> */}
          {/* <Grid.Column> */}
          <a
            href="https://zhiwenfan.github.io/"
            target="_blank"
            rel="noreferrer"
          >
            Zhiwen Fan<sup>1</sup>
          </a>
          {/* </Grid.Column> */}
          {/* </Grid.Row> */}
        </Grid>
        <Grid
          container
          stackable
          verticalAlign="middle"
          columns={4}
          textAlign="center"
          style={{
            fontSize: "1em",
            fontWeight: "normal",
          }}
        >
          {/* <Grid.Row> */}
          {/* <Grid.Column> */}
          <a
            href="https://www.humphreyshi.com/"
            target="_blank"
            rel="noreferrer"
          >
            Humphrey Shi<sup>2,3,4</sup>
          </a>
          {/* </Grid.Column> */}
          {/* <Grid.Column> */}
          <a
            href="https://express.adobe.com/page/CAdrFMJ9QeI2y/"
            target="_blank"
            rel="noreferrer"
          >
            Zhangyang Wang<sup>1</sup>
          </a>
          {/* </Grid.Column> */}
          {/* </Grid.Row> */}
          <div>
            <sup>1</sup> The University of Texas at Austin, <sup>2</sup>UIUC,{" "}
            <sup>3</sup>University of Oregon, <sup>4</sup>Picsart AI Research
          </div>
          <br />
          <div>*denotes equal contribution</div>
        </Grid>
      </Container>
      <Grid
        container
        stackable
        verticalAlign="middle"
        columns={2}
        width={10}
        textAlign="center"
        style={{
          fontSize: "1.2em",
          fontWeight: "normal",
          marginTop: "0.5em",
          marginBottom: "0.5em",
        }}
      >
        <Grid.Row>
          <Grid.Column columns={2}>
            <Button primary icon>
              <a
                href="https://arxiv.org/abs/2204.00928"
                target="_blank"
                rel="noreferrer"
                style={{
                  color: "white",
                }}
              >
                <Icon name="book" />
                {" Paper"}
              </a>
            </Button>
            <Button primary icon>
              <a
                href="https://github.com/Ir1d/SinNeRF"
                target="_blank"
                rel="noreferrer"
                style={{
                  color: "white",
                }}
              >
                <Icon name="github" />
                {" Code"}
              </a>
            </Button>
          </Grid.Column>
        </Grid.Row>
      </Grid>
      <Container text>
        <Abstract mobile={mobile} />
        {/* <Header as="h2" textAlign="center" style={subsection_style}>
          Video
        </Header>
        <video
          src={video_website}
          width="100%"
          loop="loop"
          autoplay=""
          controls
        ></video> */}
        <Header as="h2" textAlign="center" style={subsection_style}>
          Method Overview
        </Header>
        <Image src={overview} style={subsection_style}></Image>
        <p>
          An overview of our SinNeRF, where we synthesize patches from the
          reference view and unseen views. We train this semi-supervised
          framework via ground truth color and depth labels of the reference
          view and pseudo labels on unseen views. We use image warping to obtain
          geometry pseudo labels and utilize adversarial training as well as a
          pre-trained ViT for semantic pseudo labels.
        </p>
        <Results />
        <Comparison />
        <Header as="h2" textAlign="center" style={subsection_style}>
          Citation
        </Header>
        <p style={{ paddingBottom: "0.5em", overflow: "auto" }}>
          If you want to cite our work, please use:
          <pre>
            <code>
              {`@InProceedings\{Xu_2022_SinNeRF,
    author = \{Xu, Dejia and Jiang, Yifan and Wang, Peihao and Fan, Zhiwen and Shi, Humphrey and Wang, Zhangyang\},
    title = \{SinNeRF: Training Neural Radiance Fields on Complex Scenes from a Single Image\},
    journal={arXiv preprint arXiv:2204.00928},
    year={2022}
\}`}
            </code>
          </pre>
        </p>
      </Container>
    </div>
  </ResponsiveContainer>
);

export default HomepageLayout;
