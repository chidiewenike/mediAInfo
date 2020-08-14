import React from 'react';
import './TextInput.css';
import Progress from 'react-progressbar';

class TextInp extends React.Component {
  constructor(props) {
    super(props);
    this.state = {value: '', entity: '', summary: '', sentiment: '', bias: '', sentences: ''};
    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  handleChange(event) {
    this.setState({value: event.target.value});
  }

  handleSubmit(event) {

      const requestOptions = {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json' 
        },
        body: JSON.stringify({"Input": this.state.value})
    };
    
      fetch('http://127.0.0.1:5000/', requestOptions)
        .then(response => response.json())
        .then(answer => this.setState({entity : answer["entity"], bias : answer["bias"], sentiment : answer["sentiment"], summary : answer["summary"], sentences : answer["sentences"]}));//(this.entity = answer["entity"], this.bias = answer["bias"], this.sentiment = answer["sentiment"], this.summary = answer["summary"])));

    console.log(this.test)
    event.preventDefault();
  }

  render() {
    return (
      <form onSubmit={this.handleSubmit}>
        <div><img src="https://i.imgur.com/MqhuYlm.png"/></div>
        {/* <h1 className="header">Media Bias</h1> */}
        <label>
          <textarea className="box1" value={this.state.value} onChange={this.handleChange} placeholder="Enter text for analysis here..."/>
        </label>
        <input className="button" type="submit" value="Analyze Text" />
        <div className="bias">
          <h2>Overall Ratings</h2>
          <h3>Bias/Factual</h3>
          <div className="details" >
            <p>{this.state.bias}</p>
        </div>
        <h3>Sentiment</h3>
        <div className="details" >
            <p>{this.state.sentiment}</p>
        </div>        
      </div>
      <div className="entity">
          <h2>References</h2>
          <div className="details" >
            <p>{this.state.entity}</p>
        </div>
      </div>
      <div className="summary">
          <h2>Summary</h2>
          <div className="details" >
            <p>{this.state.summary}</p>
        </div>
      </div>      
      <div className="sentiment">
          <h2>Sentence-By-Sentence Sentiment Analysis</h2>
          <div className="details" >
            <p>{this.state.sentences}</p>
        </div>
      </div>
      </form>
    );
  }
}

export default TextInp;